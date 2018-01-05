/*
 * Copyright (c) 2018 by Andrew Charneski.
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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.ComponentTest;

/**
 * Attempt to trigger software bug
 */
public class BugReproduction extends CudnnLayerTestBase {
  
  /**
   * Instantiates a new Irregular run float.
   */
  public BugReproduction() {
    this.validateDifferentials = false;
  }
  
  @Override
  public NNLayer getReferenceLayer() {
    return null;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {30, 30, 512}
    };
  }
  
  @Override
  public int[][] getPerfDims() {
    return getInputDims();
  }
  
  @Override
  protected ComponentTest<ToleranceStatistics> getJsonTester() {
    return null; // new SerializationTest();
  }
  
  @Override
  public NNLayer getLayer(int[][] inputSize) {
    PipelineNetwork model = new PipelineNetwork();


//    //  model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
//    model.add(new AssertDimensionsLayer(224, 224, 3));
//    model.add(new ImgZeroPaddingLayer(1, 1));
//    //  model.add(Convolution2D(64, 3, 3, activation='relu'))
//    model.add(new ConvolutionLayer(3, 3, 3, 64)
//                .setPaddingXY(0, 0)
//                .setWeightsLog(-2)
//                .explode());
//    model.add(new ImgBandBiasLayer(64)
//                .setWeightsLog(-2));
//    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
//    //  model.add(ZeroPadding2D((1,1)))
//    model.add(new ImgZeroPaddingLayer(1, 1));
//    //  model.add(Convolution2D(64, 3, 3, activation='relu'))
//    model.add(new ConvolutionLayer(3, 3, 64, 64)
//                .setPaddingXY(0, 0)
//                .setWeightsLog(-2)
//                .explode());
//    model.add(new ImgBandBiasLayer(64)
//                .setWeightsLog(-2));
//    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
//    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
//    model.add(new PoolingLayer()
//                .setMode(PoolingLayer.PoolingMode.Max)
//                .setWindowXY(2, 2)
//                .setStrideXY(2, 2));
//
//    //  model.add(ZeroPadding2D((1,1)))
//    model.add(new ImgZeroPaddingLayer(1, 1));
//    //  model.add(Convolution2D(128, 3, 3, activation='relu'))
//    model.add(new ConvolutionLayer(3, 3, 64, 128)
//                .setPaddingXY(0, 0)
//                .setWeightsLog(-2)
//                .explode());
//    model.add(new ImgBandBiasLayer(128)
//                .setWeightsLog(-2));
//    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
//    //  model.add(ZeroPadding2D((1,1)))
//    model.add(new ImgZeroPaddingLayer(1, 1));
//    //  model.add(Convolution2D(128, 3, 3, activation='relu'))
//    model.add(new ConvolutionLayer(3, 3, 128, 128)
//                .setPaddingXY(0, 0)
//                .setWeightsLog(-2)
//                .explode());
//    model.add(new ImgBandBiasLayer(128)
//                .setWeightsLog(-2));
//    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
//    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
//    model.add(new PoolingLayer()
//                .setMode(PoolingLayer.PoolingMode.Max)
//                .setWindowXY(2, 2)
//                .setStrideXY(2, 2));
//    //
//    //  model.add(ZeroPadding2D((1,1)))
//    model.add(new ImgZeroPaddingLayer(1, 1));
//    //  model.add(Convolution2D(256, 3, 3, activation='relu'))
//    model.add(new ConvolutionLayer(3, 3, 128, 256)
//                .setPaddingXY(0, 0)
//                .setWeightsLog(-2)
//                .explode());
//    model.add(new ImgBandBiasLayer(256)
//                .setWeightsLog(-2));
//    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
//    //  model.add(ZeroPadding2D((1,1)))
//    model.add(new ImgZeroPaddingLayer(1, 1));
//    //  model.add(Convolution2D(256, 3, 3, activation='relu'))
//    model.add(new ConvolutionLayer(3, 3, 256, 256)
//                .setPaddingXY(0, 0)
//                .setWeightsLog(-2)
//                .explode());
//    model.add(new ImgBandBiasLayer(256)
//                .setWeightsLog(-2));
//    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
//    //  model.add(ZeroPadding2D((1,1)))
//    model.add(new ImgZeroPaddingLayer(1, 1));
//    //  model.add(Convolution2D(256, 3, 3, activation='relu'))
//    model.add(new ConvolutionLayer(3, 3, 256, 256)
//                .setPaddingXY(0, 0)
//                .setWeightsLog(-2)
//                .explode());
//    model.add(new ImgBandBiasLayer(256)
//                .setWeightsLog(-2));
//    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
//    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
//    model.add(new PoolingLayer()
//                .setMode(PoolingLayer.PoolingMode.Max)
//                .setWindowXY(2, 2)
//                .setStrideXY(2, 2));
//    //
//    //  model.add(ZeroPadding2D((1,1)))
//    model.add(new ImgZeroPaddingLayer(1, 1));
//    //  model.add(Convolution2D(512, 3, 3, activation='relu'))
//    model.add(new ConvolutionLayer(3, 3, 256, 512)
//                .setPaddingXY(0, 0)
//                .setWeightsLog(-2)
//                .explode());
//    model.add(new ImgBandBiasLayer(512)
//                .setWeightsLog(-2));
//    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
//    //  model.add(ZeroPadding2D((1,1)))
//    model.add(new ImgZeroPaddingLayer(1, 1));
//    //  model.add(Convolution2D(512, 3, 3, activation='relu'))



    model.add(new ConvolutionLayer(3, 3, 512, 512)
                .setPaddingXY(0, 0)
                .setWeightsLog(-2)
                .explode());
    model.add(new ImgBandBiasLayer(512)
                .setWeightsLog(-2));
    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    //  model.add(ZeroPadding2D((1,1)))
    model.add(new ImgZeroPaddingLayer(1, 1));
    //  model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(new ConvolutionLayer(3, 3, 512, 512)
                .setPaddingXY(0, 0)
                .setWeightsLog(-2)
                .explode());
    model.add(new ImgBandBiasLayer(512)
                .setWeightsLog(-2));
    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(new PoolingLayer()
                .setMode(PoolingLayer.PoolingMode.Max)
                .setWindowXY(2, 2)
                .setStrideXY(2, 2));
    //
    //  model.add(ZeroPadding2D((1,1)))
    model.add(new ImgZeroPaddingLayer(1, 1));
    //  model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(new ConvolutionLayer(3, 3, 512, 512)
                .setPaddingXY(0, 0)
                .setWeightsLog(-2)
                .explode());
    model.add(new ImgBandBiasLayer(512)
                .setWeightsLog(-2));
    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    //  model.add(ZeroPadding2D((1,1)))
    model.add(new ImgZeroPaddingLayer(1, 1));
    //  model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(new ConvolutionLayer(3, 3, 512, 512)
                .setPaddingXY(0, 0)
                .setWeightsLog(-2)
                .explode());
    model.add(new ImgBandBiasLayer(512)
                .setWeightsLog(-2));
    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    //  model.add(ZeroPadding2D((1,1)))
    model.add(new ImgZeroPaddingLayer(1, 1));
    //  model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(new ConvolutionLayer(3, 3, 512, 512)
                .setPaddingXY(0, 0)
                .setWeightsLog(-2)
                .explode());
    model.add(new ImgBandBiasLayer(512)
                .setWeightsLog(-2));
    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(new PoolingLayer()
                .setMode(PoolingLayer.PoolingMode.Max)
                .setWindowXY(2, 2)
                .setStrideXY(2, 2));
    //
    //  model.add(Flatten())
    //  model.add(Dense(4096, activation='relu'))
    model.add(new com.simiacryptus.mindseye.layers.java.FullyConnectedLayer(new int[]{25088}, new int[]{4096})
                .setWeightsLog(-2)
                .setName("fullyconnected_32"));
    model.add(new BiasLayer(4096)
                .setWeightsLog(-2));
    //  model.add(Dropout(0.5))
    //model.add(new DropoutNoiseLayer(0.5));
    //  model.add(Dense(4096, activation='relu'))
    model.add(new com.simiacryptus.mindseye.layers.java.FullyConnectedLayer(new int[]{4096}, new int[]{4096})
                .setWeightsLog(-2));
    model.add(new BiasLayer(4096)
                .setWeightsLog(-2));
    //  model.add(Dropout(0.5))
    //model.add(new DropoutNoiseLayer(0.5));
    //  model.add(Dense(1000, activation='softmax'))
    model.add(new FullyConnectedLayer(new int[]{4096}, new int[]{1000})
                .setWeightsLog(-2)
                .setName("fullyconnected_36"));
    model.add(new BiasLayer(1000)
                .setWeightsLog(-2)
                .setName("bias_36"));
    model.add(new SoftmaxActivationLayer());

    return model;
  }
  
  @Override
  protected Class<?> getTargetClass() {
    return ConvolutionLayer.class;
  }
}
