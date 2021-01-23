/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <fstream>

#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/gradients/grad_test_helper.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;

Status MNISTForwardModel(AbstractContext* ctx,
                         absl::Span<AbstractTensorHandle* const> inputs,
                         absl::Span<AbstractTensorHandle*> outputs) {
  // inputs = `[X, W1, W2, y_labels]`

  std::vector<AbstractTensorHandle*> temp_outputs(1);
  // `mm_out_1 = tf.matmul(X,W1)`
  TF_RETURN_IF_ERROR(ops::MatMul(ctx, {inputs[0], inputs[1]},
                                 absl::MakeSpan(temp_outputs), "MNIST_MatMul_1",
                                 /*transpose_a=*/false,
                                 /*transpose_b=*/false));
  // `hidden_layer = tf.nn.relu(mm_out_1)`
  TF_RETURN_IF_ERROR(ops::Relu(ctx, {temp_outputs[0]},
                               absl::MakeSpan(temp_outputs), "MNIST_Relu"));
  // `scores = tf.matmul(hidden_layer,W2)`
  TF_RETURN_IF_ERROR(ops::MatMul(ctx, {temp_outputs[0], inputs[2]},
                                 absl::MakeSpan(temp_outputs), "MNIST_MatMul_2",
                                 /*transpose_a=*/false, /*transpose_b=*/false));
  // `loss_val, backprop =
  // tf.nn.sparse_softmax_cross_entropy_with_logits(scores, y_labels)`.
  // Note that this is just a psuedo-code. The real python code takes `labels`
  // before `scores`. In addition, we only need `loss_val`.
  temp_outputs.resize(2);
  TF_RETURN_IF_ERROR(ops::SparseSoftmaxCrossEntropyWithLogits(
      ctx, {temp_outputs[0], inputs[3]}, absl::MakeSpan(temp_outputs),
      "MNIST_SparseSoftmaxCrossEntropyWithLogits"));

  temp_outputs[1]->Unref();

  outputs[0] = temp_outputs[0];
  return Status::OK();
}

Status MNISTGradModel(AbstractContext* ctx,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs) {
  // inputs = `[X, W1, W2, y_labels]`

  Tape tape(/*persistent=*/false);
  for (size_t i{1}; i < inputs.size() - 1; ++i) {
    // We don't watch `X` and `y_labels`.
    tape.Watch(inputs[i]);
  }

  GradientRegistry registry;
  TF_RETURN_IF_ERROR(registry.Register("AddV2", AddRegisterer));
  TF_RETURN_IF_ERROR(registry.Register("MatMul", MatMulRegisterer));
  TF_RETURN_IF_ERROR(registry.Register("Relu", ReluRegisterer));
  TF_RETURN_IF_ERROR(
      registry.Register("SparseSoftmaxCrossEntropyWithLogits",
                        SparseSoftmaxCrossEntropyWithLogitsRegisterer));

  AbstractContextPtr tape_ctx(new TapeContext(ctx, &tape, registry));
  std::vector<AbstractTensorHandle*> temp_outputs(1);
  TF_RETURN_IF_ERROR(
      MNISTForwardModel(tape_ctx.get(), inputs, absl::MakeSpan(temp_outputs)));

  TF_RETURN_IF_ERROR(
      tape.ComputeGradient(ctx, /*targets=loss_val*/ temp_outputs,
                           /*sources={W1,W2}*/ {inputs[1], inputs[2]},
                           /*output_gradients=*/{}, outputs));

  UnrefTensorHandles(&temp_outputs);
  return Status::OK();
}

Status MakeMNISTInput(AbstractContext* ctx,
                      absl::Span<const absl::Span<const float>> raw_inputs,
                      absl::Span<const int> labels,
                      absl::Span<const absl::Span<const int64_t>> dim_inputs,
                      absl::Span<AbstractTensorHandle*> outputs) {
  for (int i{}; i < 3; ++i) {
    auto raw_input = raw_inputs[i];
    auto dim_input = dim_inputs[i];
    TF_RETURN_IF_ERROR(TestTensorHandleWithDimsFloat(
        ctx, const_cast<float*>(raw_input.data()),
        const_cast<int64_t*>(dim_input.data()), dim_input.size(), &outputs[i]));
  }
  {
    auto dim_input = dim_inputs[3];
    TF_RETURN_IF_ERROR(TestTensorHandleWithDimsInt(
        ctx, const_cast<int*>(labels.data()),
        const_cast<int64_t*>(dim_input.data()), dim_input.size(), &outputs[3]));
  }
  return Status::OK();
}

int ReverseInt(int i) {
  unsigned char c1, c2, c3, c4;
  c1 = (i & 255);
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) +
         (static_cast<int>(c3) << 8) + c4;
}

void ReadMNISTHeader(std::ifstream* file, int* value) {
  file->read(reinterpret_cast<char*>(value), sizeof(*value));
  *value = ReverseInt(*value);
}

Status ReadMNISTImageData(AbstractContext* ctx, const std::string& path,
                          std::vector<AbstractTensorHandle*>* images,
                          int* image_size) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) return Status(error::FAILED_PRECONDITION, path);

  int magic_number = 0;
  int number_of_images = 0;
  int n_rows = 0;
  int n_cols = 0;

  ReadMNISTHeader(&file, &magic_number);
  if (magic_number != 2051) {
    return Status(error::INVALID_ARGUMENT, std::to_string(magic_number));
  }
  ReadMNISTHeader(&file, &number_of_images);
  ReadMNISTHeader(&file, &n_rows);
  ReadMNISTHeader(&file, &n_cols);

  int image_s = n_cols * n_rows;
  if (image_size) *image_size = image_s;

  std::unique_ptr<unsigned char[]> image_data(new unsigned char[image_s]);
  std::unique_ptr<float[]> tensor_data(new float[image_s]);
  int64_t dims[] = {1, image_s};

  images->resize(number_of_images);
  for (int i = 0; i < number_of_images; i++) {
    file.read(reinterpret_cast<char*>(image_data.get()), image_s);

    for (int i{}; i < image_s; ++i) {
      tensor_data[i] = image_data[i];
    }
    TF_RETURN_IF_ERROR(TestTensorHandleWithDimsFloat(ctx, tensor_data.get(),
                                                     dims, 2, &((*images)[i])));
  }

  return Status::OK();
}

Status ReadMNISTLabelData(AbstractContext* ctx, const std::string& path,
                          std::vector<AbstractTensorHandle*>* labels) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) return Status(error::FAILED_PRECONDITION, path);

  int magic_number = 0;
  int number_of_labels = 0;

  ReadMNISTHeader(&file, &magic_number);
  if (magic_number != 2049) {
    return Status(error::INVALID_ARGUMENT, std::to_string(magic_number));
  }
  ReadMNISTHeader(&file, &number_of_labels);

  labels->resize(number_of_labels);
  unsigned char label_data = 0;
  int tensor_data = 0;
  int64_t dims[] = {1};

  for (int i = 0; i < number_of_labels; i++) {
    file.read(reinterpret_cast<char*>(&label_data), 1);

    tensor_data = label_data;
    TF_RETURN_IF_ERROR(TestTensorHandleWithDimsInt(ctx, &tensor_data, dims, 1,
                                                   &((*labels)[i])));
  }

  return Status::OK();
}

Status ReadMNISTData(AbstractContext* ctx, const std::string& dir,
                     std::vector<AbstractTensorHandle*>* train_images,
                     std::vector<AbstractTensorHandle*>* train_labels,
                     std::vector<AbstractTensorHandle*>* test_images,
                     std::vector<AbstractTensorHandle*>* test_labels,
                     int* image_size) {
  constexpr char train_images_path[] = "train-images-idx3-ubyte";
  constexpr char train_labels_path[] = "train-labels-idx1-ubyte";
  constexpr char test_images_path[] = "t10k-images-idx3-ubyte";
  constexpr char test_labels_path[] = "t10k-labels-idx1-ubyte";

  TF_RETURN_IF_ERROR(ReadMNISTImageData(
      ctx, io::JoinPath(dir, train_images_path), train_images, image_size));
  TF_RETURN_IF_ERROR(ReadMNISTLabelData(
      ctx, io::JoinPath(dir, train_labels_path), train_labels));
  TF_RETURN_IF_ERROR(ReadMNISTImageData(
      ctx, io::JoinPath(dir, test_images_path), test_images, nullptr));
  TF_RETURN_IF_ERROR(ReadMNISTLabelData(
      ctx, io::JoinPath(dir, test_labels_path), test_labels));

  return Status::OK();
}

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    status_ = StatusFromTF_Status(status.get());
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

    {
      AbstractContext* ctx_raw = nullptr;
      status_ =
          BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
      ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
      immediate_execution_ctx_.reset(ctx_raw);
    }

    // Computing numerical gradients with TensorFloat-32 is numerically
    // unstable. Some forward pass tests also fail with TensorFloat-32 due to
    // low tolerances
    enable_tensor_float_32_execution(false);
  }

  AbstractContextPtr immediate_execution_ctx_;
  Status status_;

 public:
  bool UseMlir() const { return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const { return std::get<2>(GetParam()); }
};

TEST_P(CppGradients, TestMNISTGrad) {
  if (UseFunction()) {
    // TODO(b/168850692): Enable this.
    GTEST_SKIP() << "Can't take gradient of "
                    "SparseSoftmaxCrossEntropyWithLogits in tracing mode.";
  }

  std::vector<AbstractTensorHandle*> mnist_inputs(4);
  status_ = MakeMNISTInput(immediate_execution_ctx_.get(),
                           {/*X*/ {1.0f, 2.0f, 3.0f, 4.0f},
                            /*W1*/ {-1.0f, 10.0f, .5f, 1.0f},
                            /*W2*/ {.1f, .2f, .3f, -.5f}},
                           /*y_labels*/ {1, 1}, {{2, 2}, {2, 2}, {2, 2}, {2}},
                           absl::MakeSpan(mnist_inputs));
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  std::vector<AbstractTensorHandle*> outputs(2);
  status_ = RunModel(MNISTGradModel, immediate_execution_ctx_.get(),
                     mnist_inputs, absl::MakeSpan(outputs), UseFunction());
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  double abs_error = 1e-3;
  // dW1
  ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[0], {0.0f, 3.2f, 0.0f, 4.8f},
                                           /*dims*/ {2, 2}, abs_error));
  // dW2
  ASSERT_NO_FATAL_FAILURE(CheckTensorValue(outputs[1],
                                           {0.0f, 0.0f, 46.0f, -46.0f},
                                           /*dims*/ {2, 2}, abs_error));
  UnrefTensorHandles(&mnist_inputs);
  UnrefTensorHandles(&outputs);
}

TEST_P(CppGradients, TestMNISTTraining) {
  if (UseFunction()) {
    // TODO(b/168850692): Enable this.
    GTEST_SKIP() << "Can't take gradient of "
                    "SparseSoftmaxCrossEntropyWithLogits in tracing mode.";
  }

  auto dataset_dir =
      std::getenv("TF_CPP_GRADIENTS_MNIST_GRAD_TEST_DATASET_DIR");
  if (!dataset_dir) GTEST_SKIP();

  std::vector<AbstractTensorHandle*> train_images;
  std::vector<AbstractTensorHandle*> train_labels;
  std::vector<AbstractTensorHandle*> test_images;
  std::vector<AbstractTensorHandle*> test_labels;
  int image_size;

  status_ =
      ReadMNISTData(immediate_execution_ctx_.get(), dataset_dir, &train_images,
                    &train_labels, &test_images, &test_labels, &image_size);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
