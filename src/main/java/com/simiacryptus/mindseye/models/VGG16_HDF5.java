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
package com.simiacryptus.mindseye.models;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.AssertDimensionsLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.concurrent.Callable;

/**
 * Details about this network architecture can be found in the following arXiv paper: Very Deep Convolutional Networks
 * for Large-Scale Image Recognition K. Simonyan, A. Zisserman arXiv:1409.1556 Please cite the paper if you use the
 * models.
 */
class VGG16_HDF5 extends VGG16 implements DemoableNetworkFactory, HasHDF5 {
  private static final Logger log = LoggerFactory.getLogger(Hdf5Archive.class);
  private final Hdf5Archive hdf5;
  private volatile Layer network;
  
  /**
   * Instantiates a new Vgg 16 hdf 5.
   *
   * @param hdf5 the hdf 5
   */
  public VGG16_HDF5(Hdf5Archive hdf5) {this.hdf5 = hdf5;}
  
  /**
   * Gets network.
   *
   * @return the network
   */
  public Layer getNetwork() {
    if (null == network) {
      synchronized (this) {
        if (null == network) {
          network = build();
        }
      }
    }
    return network;
  }
  
  @javax.annotation.Nonnull
  @Override
  public Layer build(@javax.annotation.Nonnull NotebookOutput output) {
    try {
      return new Callable<Layer>() {
        @Nullable
        Tensor prototype = new Tensor(224, 224, 3);
        int cnt = 1;
        @javax.annotation.Nonnull
        int[] convolutionOrder = {2, 3, 0, 1};
        @javax.annotation.Nonnull
        int[] fullyconnectedOrder = {1, 0};
        @javax.annotation.Nonnull
        PipelineNetwork model = new PipelineNetwork();
        @javax.annotation.Nonnull
        Precision precision = Precision.Float;
        
        @javax.annotation.Nonnull
        @Override
        public Layer call() throws Exception {
          //  model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
          output.code(() -> {
            add(new AssertDimensionsLayer(224, 224, 3));
          });
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(64, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 3, 64)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_1")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(64)
              .set((hdf5.readDataSet("param_1", "layer_1"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(64, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 64, 64)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_3")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(64)
              .set((hdf5.readDataSet("param_1", "layer_3"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(MaxPooling2D((2,2), strides=(2,2)))
          output.code(() -> {
            add(new PoolingLayer()
              .setMode(PoolingLayer.PoolingMode.Max)
              .setWindowXY(2, 2)
              .setStrideXY(2, 2));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(128, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 64, 128)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_6")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(128)
              .set((hdf5.readDataSet("param_1", "layer_6"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(128, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 128, 128)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_8")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(128)
              .set((hdf5.readDataSet("param_1", "layer_8"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(MaxPooling2D((2,2), strides=(2,2)))
          output.code(() -> {
            add(new PoolingLayer()
              .setMode(PoolingLayer.PoolingMode.Max)
              .setWindowXY(2, 2)
              .setStrideXY(2, 2));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(256, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 128, 256)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_11")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(256)
              .set((hdf5.readDataSet("param_1", "layer_11"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(256, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 256, 256)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_13")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(256)
              .set((hdf5.readDataSet("param_1", "layer_13"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(256, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 256, 256)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_15")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(256)
              .set((hdf5.readDataSet("param_1", "layer_15"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(MaxPooling2D((2,2), strides=(2,2)))
          output.code(() -> {
            add(new PoolingLayer()
              .setMode(PoolingLayer.PoolingMode.Max)
              .setWindowXY(2, 2)
              .setStrideXY(2, 2));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(512, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 256, 512)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_18")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(512)
              .set((hdf5.readDataSet("param_1", "layer_18"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(512, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 512, 512)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_20")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(512)
              .set((hdf5.readDataSet("param_1", "layer_20"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(512, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 512, 512)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_22")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(512)
              .set((hdf5.readDataSet("param_1", "layer_22"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(MaxPooling2D((2,2), strides=(2,2)))
          output.code(() -> {
            add(new PoolingLayer()
              .setMode(PoolingLayer.PoolingMode.Max)
              .setWindowXY(2, 2)
              .setStrideXY(2, 2));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(512, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 512, 512)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_25")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(512)
              .set((hdf5.readDataSet("param_1", "layer_25"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(512, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 512, 512)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_27")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(512)
              .set((hdf5.readDataSet("param_1", "layer_27"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(ZeroPadding2D((1,1)))
          output.code(() -> {
            add(new ImgZeroPaddingLayer(1, 1));
          });
          //  model.add(Convolution2D(512, 3, 3, activation='relu'))
          output.code(() -> {
            add(new ConvolutionLayer(3, 3, 512, 512)
              .setPaddingXY(0, 0)
              .set(hdf5.readDataSet("param_0", "layer_29")
                .permuteDimensions(convolutionOrder))
            );
          });
          output.code(() -> {
            add(new ImgBandBiasLayer(512)
              .set((hdf5.readDataSet("param_1", "layer_29"))));
          });
          output.code(() -> {
            add(new ActivationLayer(ActivationLayer.Mode.RELU));
          });
          //  model.add(MaxPooling2D((2,2), strides=(2,2)))
          output.code(() -> {
            add(new PoolingLayer()
              .setMode(PoolingLayer.PoolingMode.Max)
              .setWindowXY(2, 2)
              .setStrideXY(2, 2));
          });
          //  model.add(Flatten())
          //  model.add(Dense(4096, activation='relu'))
          output.code(() -> {
            add(new FullyConnectedLayer(new int[]{25088}, new int[]{4096})
              .set(hdf5.readDataSet("param_0", "layer_32")
                .permuteDimensions(fullyconnectedOrder)));
          });
          output.code(() -> {
            add(new BiasLayer(4096)
              .set((hdf5.readDataSet("param_1", "layer_32"))));
          });
          //  model.add(Dropout(0.5))
          //model.add(new DropoutNoiseLayer(0.5));
          //  model.add(Dense(4096, activation='relu'))
          output.code(() -> {
            add(new FullyConnectedLayer(new int[]{4096}, new int[]{4096})
              .set(hdf5.readDataSet("param_0", "layer_34")
                .permuteDimensions(fullyconnectedOrder))
            );
          });
          output.code(() -> {
            add(new BiasLayer(4096)
              .set((hdf5.readDataSet("param_1", "layer_34"))));
          });
          //  model.add(Dropout(0.5))
          //model.add(new DropoutNoiseLayer(0.5));
          //  model.add(Dense(1000, activation='softmax'))
          output.code(() -> {
            add(new FullyConnectedLayer(new int[]{4096}, new int[]{1000})
              .set(hdf5.readDataSet("param_0", "layer_36")
                .permuteDimensions(fullyconnectedOrder)));
          });
          output.code(() -> {
            add(new BiasLayer(1000)
              .set((hdf5.readDataSet("param_1", "layer_36"))));
          });
          output.code(() -> {
            add(new SoftmaxActivationLayer());
          });
          output.code(() -> {
            model.visitLayers(layer -> {
              if (layer instanceof MultiPrecision) {
                ((MultiPrecision) layer).setPrecision(precision);
              }
            });
            return precision;
          });
          if (null != prototype) prototype.freeRef();
          prototype = null;
          return model;
        }
    
        protected void add(Layer layer) {
          name(layer);
          if (layer instanceof Explodable) {
            DAGNetwork explode = ((Explodable) layer).explode();
            explode.visitNodes(node -> name(node.getLayer()));
            log.info(String.format("Exploded %s to %s (%s nodes)", layer.getName(), explode.getClass().getSimpleName(), explode.getNodes().size()));
            add(explode);
          }
          else {
            int numberOfParameters = layer.state().stream().mapToInt(x -> x.length).sum();
            model.add(layer);
            @javax.annotation.Nonnull int[] prev_dimensions = prototype.getDimensions();
            NNResult eval = layer.eval(prototype);
            TensorList data = eval.getData();
            if (null != prototype) prototype.freeRef();
            prototype = data.get(0);
            eval.freeRef();
            data.freeRef();
            @javax.annotation.Nonnull int[] new_dimensions = prototype.getDimensions();
            log.info(String.format("Added layer #%d: %s; %s params, dimensions %s (%s) -> %s (%s)", //
              cnt++, layer, numberOfParameters, //
              Arrays.toString(prev_dimensions), Tensor.dim(prev_dimensions), //
              Arrays.toString(new_dimensions), Tensor.dim(new_dimensions)));
          }
        }
      }.call();
    } catch (@javax.annotation.Nonnull final RuntimeException e) {
      throw e;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
  
  private void name(final Layer layer) {
    if (layer.getName().contains(layer.getId().toString())) {
      if (layer instanceof ConvolutionLayer) {
        layer.setName(layer.getClass().getSimpleName() + ((ConvolutionLayer) layer).getConvolutionParams());
      }
      else if (layer instanceof SimpleConvolutionLayer) {
        layer.setName(String.format("%s: %s", layer.getClass().getSimpleName(),
          Arrays.toString(((SimpleConvolutionLayer) layer).getKernelDimensions())));
      }
      else if (layer instanceof FullyConnectedLayer) {
        layer.setName(String.format("%s:%sx%s",
          layer.getClass().getSimpleName(),
          Arrays.toString(((FullyConnectedLayer) layer).inputDims),
          Arrays.toString(((FullyConnectedLayer) layer).outputDims)));
      }
      else if (layer instanceof BiasLayer) {
        layer.setName(String.format("%s:%s",
          layer.getClass().getSimpleName(),
          ((BiasLayer) layer).bias.length));
      }
    }
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
