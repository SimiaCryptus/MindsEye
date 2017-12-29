/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.deeplearning4j.nn.modelimport.keras.preprocessors;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.preprocessor.BaseInputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Generic reshape preprocessor
 *
 * @author Max Pumperla
 */
public class ReshapePreprocessor extends BaseInputPreProcessor {
  
  private int[] inputShape;
  private int[] targetShape;
  private boolean hasMiniBatchDimension = false;
  private int miniBatchSize;
  
  public ReshapePreprocessor(int[] inputShape, int[] targetShape) {
    this.inputShape = inputShape;
    this.targetShape = targetShape;
  }
  
  
  private static int prod(int[] array) {
    int prod = 1;
    for (int i : array) {
      prod *= i;
    }
    return prod;
  }
  
  private static int[] prependMiniBatchSize(int[] shape, int miniBatchSize) {
    int shapeLength = shape.length;
    int[] miniBatchShape = new int[shapeLength + 1];
    for (int i = 0; i < miniBatchShape.length; i++) {
      if (i == 0) {
        miniBatchShape[0] = miniBatchSize;
      }
      else {
        miniBatchShape[i] = shape[i - 1];
      }
    }
    return miniBatchShape;
  }
  
  @Override
  public INDArray preProcess(INDArray input, int miniBatchSize) {
    // the target shape read from a keras config does not have mini-batch size
    // included. We prepend it here dynamically.
    if (targetShape.length + 1 == input.shape().length) {
      targetShape = prependMiniBatchSize(targetShape, miniBatchSize);
      inputShape = prependMiniBatchSize(inputShape, miniBatchSize);
      this.hasMiniBatchDimension = true;
      this.miniBatchSize = miniBatchSize;
    }
    if (this.miniBatchSize != miniBatchSize) {
      targetShape = prependMiniBatchSize(ArrayUtils.subarray(targetShape, 1, targetShape.length), miniBatchSize);
      inputShape = prependMiniBatchSize(ArrayUtils.subarray(inputShape, 1, targetShape.length), miniBatchSize);
      this.miniBatchSize = miniBatchSize;
    }
    if (prod(input.shape()) == prod((targetShape))) {
      return input.reshape(this.targetShape);
    }
    else {
      throw new IllegalStateException("Input shape " + Arrays.toString(input.shape())
        + " and output shape" + Arrays.toString(inputShape) + " do not match");
    }
  }
  
  @Override
  public INDArray backprop(INDArray output, int miniBatchSize) {
    if (!Arrays.equals(targetShape, output.shape())) {
      throw new IllegalStateException("Unexpected output shape" + Arrays.toString(output.shape())
        + " (expected to be " + Arrays.toString(targetShape) + ")");
    }
    if (prod(output.shape()) == prod((targetShape))) {
      return output.reshape(this.inputShape);
    }
    else {
      throw new IllegalStateException("Output shape" + Arrays.toString(output.shape())
        + " and input shape" + Arrays.toString(targetShape) + " do not match");
    }
  }
  
  @Override
  public InputType getOutputType(InputType inputType) throws InvalidInputTypeException {
    
    int[] shape = hasMiniBatchDimension ? targetShape : prependMiniBatchSize(targetShape, 0);
    switch (shape.length) {
      case 2:
        return InputType.feedForward(shape[1]);
      case 3:
        return InputType.recurrent(shape[1]);
      case 4:
        return InputType.convolutional(shape[2], shape[3], shape[1]);
      default:
        throw new UnsupportedOperationException(
          "Cannot infer input type for reshape array " + Arrays.toString(shape));
    }
  }
}