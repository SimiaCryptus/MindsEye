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

package com.simiacryptus.mindseye.models;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.AssertDimensionsLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Details about this network architecture can be found in the following arXiv paper:
 * <p>
 * Very Deep Convolutional Networks for Large-Scale Image Recognition K. Simonyan, A. Zisserman arXiv:1409.1556
 * <p>
 * Please cite the paper if you use the models.
 */
class VGG16_HDF5 extends VGG16 implements DemoableNetworkFactory, HasHDF5 {
  
  private volatile NNLayer network;
  
  /**
   * Gets network.
   *
   * @return the network
   */
  public NNLayer getNetwork() {
    if (null == network) {
      synchronized (this) {
        if (null == network) {
          network = build();
        }
      }
    }
    return network;
  }
  
  
  
  private static final Logger log = LoggerFactory.getLogger(Hdf5Archive.class);
  private final Hdf5Archive hdf5;
  
  /**
   * Instantiates a new Vgg 16 hdf 5.
   *
   * @param hdf5 the hdf 5
   */
  public VGG16_HDF5(Hdf5Archive hdf5) {this.hdf5 = hdf5;}
  
  @Override
  public NNLayer build(NotebookOutput output) {
    return output.code(() -> {
      PipelineNetwork model = new PipelineNetwork();
      
      int[] convolutionOrder = {2, 3, 0, 1};
      int[] fullyconnectedOrder = {1, 0};
      
      //  model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
      model.add(new AssertDimensionsLayer(224, 224, 3));
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(64, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 3, 64)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_1")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(64)
                  .set((hdf5.readDataSet("param_1", "layer_1"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(ZeroPadding2D((1,1)))
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(64, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 64, 64)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_3")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(64)
                  .set((hdf5.readDataSet("param_1", "layer_3"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(MaxPooling2D((2,2), strides=(2,2)))
      model.add(new PoolingLayer()
                  .setMode(PoolingLayer.PoolingMode.Max)
                  .setWindowXY(2, 2)
                  .setStrideXY(2, 2));
      
      //  model.add(ZeroPadding2D((1,1)))
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(128, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 64, 128)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_6")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(128)
                  .set((hdf5.readDataSet("param_1", "layer_6"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(ZeroPadding2D((1,1)))
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(128, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 128, 128)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_8")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(128)
                  .set((hdf5.readDataSet("param_1", "layer_8"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(MaxPooling2D((2,2), strides=(2,2)))
      model.add(new PoolingLayer()
                  .setMode(PoolingLayer.PoolingMode.Max)
                  .setWindowXY(2, 2)
                  .setStrideXY(2, 2));
      //
      //  model.add(ZeroPadding2D((1,1)))
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(256, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 128, 256)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_11")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(256)
                  .set((hdf5.readDataSet("param_1", "layer_11"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(ZeroPadding2D((1,1)))
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(256, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 256, 256)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_13")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(256)
                  .set((hdf5.readDataSet("param_1", "layer_13"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(ZeroPadding2D((1,1)))
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(256, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 256, 256)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_15")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(256)
                  .set((hdf5.readDataSet("param_1", "layer_15"))));
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
      model.add(new ConvolutionLayer(3, 3, 256, 512)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_18")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(512)
                  .set((hdf5.readDataSet("param_1", "layer_18"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(ZeroPadding2D((1,1)))
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(512, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 512, 512)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_20")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(512)
                  .set((hdf5.readDataSet("param_1", "layer_20"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(ZeroPadding2D((1,1)))
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(512, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 512, 512)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_22")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(512)
                  .set((hdf5.readDataSet("param_1", "layer_22"))));
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
                  .set(hdf5.readDataSet("param_0", "layer_25")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(512)
                  .set((hdf5.readDataSet("param_1", "layer_25"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(ZeroPadding2D((1,1)))
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(512, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 512, 512)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_27")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(512)
                  .set((hdf5.readDataSet("param_1", "layer_27"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(ZeroPadding2D((1,1)))
      model.add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(512, 3, 3, activation='relu'))
      model.add(new ConvolutionLayer(3, 3, 512, 512)
                  .setPaddingXY(0, 0)
                  .set(hdf5.readDataSet("param_0", "layer_29")
                           .permuteDimensions(convolutionOrder)));
      model.add(new ImgBandBiasLayer(512)
                  .set((hdf5.readDataSet("param_1", "layer_29"))));
      model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      //  model.add(MaxPooling2D((2,2), strides=(2,2)))
      model.add(new PoolingLayer()
                  .setMode(PoolingLayer.PoolingMode.Max)
                  .setWindowXY(2, 2)
                  .setStrideXY(2, 2));
      //
      //  model.add(Flatten())
      //  model.add(Dense(4096, activation='relu'))
      model.add(new FullyConnectedLayer(new int[]{25088}, new int[]{4096})
                  .set(hdf5.readDataSet("param_0", "layer_32")
                           .permuteDimensions(fullyconnectedOrder))
                  .setName("fullyconnected_32"));
      model.add(new BiasLayer(4096)
                  .set((hdf5.readDataSet("param_1", "layer_32"))));
      //  model.add(Dropout(0.5))
      //model.add(new DropoutNoiseLayer(0.5));
      //  model.add(Dense(4096, activation='relu'))
      model.add(new FullyConnectedLayer(new int[]{4096}, new int[]{4096})
                  .set(hdf5.readDataSet("param_0", "layer_34")
                           .permuteDimensions(fullyconnectedOrder)));
      model.add(new BiasLayer(4096)
                  .set((hdf5.readDataSet("param_1", "layer_34"))));
      //  model.add(Dropout(0.5))
      //model.add(new DropoutNoiseLayer(0.5));
      //  model.add(Dense(1000, activation='softmax'))
      model.add(new FullyConnectedLayer(new int[]{4096}, new int[]{1000})
                  .set(hdf5.readDataSet("param_0", "layer_36")
                           .permuteDimensions(fullyconnectedOrder))
                  .setName("fullyconnected_36"));
      model.add(new BiasLayer(1000)
                  .set((hdf5.readDataSet("param_1", "layer_36")))
                  .setName("bias_36"));
      model.add(new SoftmaxActivationLayer());
      
      return model;
    });
  }
  
  /**
   * Gets hdf 5.
   *
   * @return the hdf 5
   */
  @Override
  public Hdf5Archive getHDF5() {
    return hdf5;
  }
}
