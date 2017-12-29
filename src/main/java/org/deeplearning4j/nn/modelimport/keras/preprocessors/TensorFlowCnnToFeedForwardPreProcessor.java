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

import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;

/**
 * Specialized CnnToFeedForwardInputPreProcessor for use with
 * Convolutional layers imported from Keras using the TensorFlow
 * backend.
 *
 * @author dave@skymind.io
 */
public class TensorFlowCnnToFeedForwardPreProcessor extends CnnToFeedForwardPreProcessor {
  public TensorFlowCnnToFeedForwardPreProcessor(int height, int width, int depth) {
    super(height, width, depth);
  }

//  @JsonCreator
//  public TensorFlowCnnToFeedForwardPreProcessor(@JsonProperty("inputHeight") int inputHeight,
//                                                @JsonProperty("inputWidth") int inputWidth, @JsonProperty("numChannels") int numChannels) {
//    super(inputHeight, inputWidth, numChannels);
//  }
//
//  public TensorFlowCnnToFeedForwardPreProcessor(int inputHeight, int inputWidth) {
//    super(inputHeight, inputWidth);
//  }
//
//  public TensorFlowCnnToFeedForwardPreProcessor() {
//    super();
//  }
  
  @Override
  public INDArray preProcess(INDArray input, int miniBatchSize) {
    if (input.rank() == 2) {
      return input; //Should usually never happen
    }
        /* DL4J convolutional input:       # channels, # rows, # cols
         * TensorFlow convolutional input: # rows, # cols, # channels
         * Theano convolutional input:     # channels, # rows, # cols
         */
    INDArray permuted = input.permute(0, 2, 3, 1).dup('c'); //To: [n, h, w, c]
    
    int[] inShape = input.shape(); //[miniBatch,depthOut,outH,outW]
    int[] outShape = {inShape[0], inShape[1] * inShape[2] * inShape[3]};
    
    return permuted.reshape('c', outShape);
  }
  
  @Override
  public INDArray backprop(INDArray epsilons, int miniBatchSize) {
    if (epsilons.ordering() != 'c' || !Shape.strideDescendingCAscendingF(epsilons)) {
      epsilons = epsilons.dup('c');
    }
    
    throw new RuntimeException("NI");
//        INDArray epsilonsReshaped = epsilons.reshape('c', epsilons.size(0), inputHeight, inputWidth, numChannels);
//        return epsilonsReshaped.permute(0, 3, 1, 2);    //To [n, c, h, w]
  }
  
  @Override
  public TensorFlowCnnToFeedForwardPreProcessor clone() {
    try {
      return (TensorFlowCnnToFeedForwardPreProcessor) super.clone();
    } catch (CloneNotSupportedException e) {
      throw new RuntimeException(e);
    }
  }
}
