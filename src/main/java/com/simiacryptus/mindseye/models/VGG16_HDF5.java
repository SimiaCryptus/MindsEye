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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgZeroPaddingLayer;
import com.simiacryptus.mindseye.layers.cudnn.StochasticSamplingSubnetLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;

/**
 * Details about this network architecture can be found in the following arXiv paper: Very Deep Convolutional Networks
 * for Large-Scale Image Recognition K. Simonyan, A. Zisserman arXiv:1409.1556 Please cite the paper if you use the
 * models.
 */
public class VGG16_HDF5 extends VGG16 implements NetworkFactory, HasHDF5 {
  
  /**
   * The constant log.
   */
  protected static final Logger log = LoggerFactory.getLogger(VGG16_HDF5.class);
  /**
   * The Pipeline network.
   */
  public final PipelineNetwork pipelineNetwork = new PipelineNetwork();
  /**
   * The Hdf 5.
   */
  protected final Hdf5Archive hdf5;
  /**
   * The Convolution order.
   */
  @javax.annotation.Nonnull
  int[] convolutionOrder = {3, 2, 0, 1};
  /**
   * The Fullyconnected order.
   */
  @javax.annotation.Nonnull
  int[] fullyconnectedOrder = {1, 0};
  private PoolingLayer.PoolingMode finalPoolingMode = PoolingLayer.PoolingMode.Max;
  /**
   * The Precision.
   */
  private boolean large = true;
  private boolean dense = true;
  
  /**
   * Instantiates a new Vgg 16 hdf 5.
   *
   * @param hdf5 the hdf 5
   */
  public VGG16_HDF5(Hdf5Archive hdf5) {this.hdf5 = hdf5;}
  
  /**
   * Add.
   *
   * @param layer the layer
   */
  protected void add(@Nonnull Layer layer) {
    this.prototype = evaluatePrototype(add(layer, pipelineNetwork), this.prototype, cnt++);
  }
  
  public Layer buildNetwork() {
    prototype = new Tensor(224, 224, 3);
    phase0();
    phase1();
    phase2();
    phase3();
    return pipelineNetwork;
  }
  
  /**
   * Phase 1.
   */
  protected void phase1() {
    phase1a();
    phase1b();
    phase1c();
    phase1d();
    phase1e();
  }
  
  /**
   * Phase 1 b.
   */
  protected void phase1b() {
    addPoolingLayer(2);
    addConvolutionLayer(3, 64, 128, ActivationLayer.Mode.RELU, "layer_6");
    addConvolutionLayer(3, 128, 128, ActivationLayer.Mode.RELU, "layer_8");
  }
  
  /**
   * Phase 0.
   */
  protected void phase0() {
    add(new ImgMinSizeLayer(226, 226));
  }
  
  /**
   * Phase 1 c.
   */
  protected void phase1c() {
    addPoolingLayer(2);
    addConvolutionLayer(3, 128, 256, ActivationLayer.Mode.RELU, "layer_11");
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_13");
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_15");
  }
  
  /**
   * Phase 1 a.
   */
  protected void phase1a() {
    addConvolutionLayer(3, 3, 64, ActivationLayer.Mode.RELU, "layer_1");
    addConvolutionLayer(3, 64, 64, ActivationLayer.Mode.RELU, "layer_3");
  }
  
  /**
   * Phase 1 d.
   */
  protected void phase1d() {
    addPoolingLayer(2);
    addConvolutionLayer(3, 256, 512, ActivationLayer.Mode.RELU, "layer_18");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_20");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_22");
  }
  
  /**
   * Phase 1 e.
   */
  protected void phase1e() {
    addPoolingLayer(2);
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_25");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_27");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_29");
  }
  
  /**
   * Phase 2.
   */
  protected void phase2() {
    phase2a();
    phase2b();
  }
  
  /**
   * Phase 2 a.
   */
  protected void phase2a() {
    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
    addPoolingLayer(2);
  }
  
  /**
   * Phase 2 b.
   */
  protected void phase2b() {
    if (large) {
      add(new ImgModulusPaddingLayer(7, 7));
    }
    else {
      add(new ImgModulusPaddingLayer(-7, -7));
    }
    
    if (dense) {
      add(new ConvolutionLayer(7, 7, 512, 4096)
        .setStrideXY(1, 1)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_32")
          .reshapeCast(7, 7, 512, 4096).permuteDimensionsAndFree(0, 1, 3, 2)
        )
      );
    }
    else {
      add(new ImgModulusPaddingLayer(7, 7));
      add(new ImgReshapeLayer(7, 7, false));
      add(new ConvolutionLayer(1, 1, 25088, 4096)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_32")
          .permuteDimensionsAndFree(fullyconnectedOrder))
      );
    }
  
    add(new ImgBandBiasLayer(4096)
      .setAndFree((hdf5.readDataSet("param_1", "layer_32"))));
    add(new ActivationLayer(ActivationLayer.Mode.RELU));
  }
  
  /**
   * Phase 3.
   */
  protected void phase3() {
    phase3a();
    phase3b();
  }
  
  /**
   * Phase 3 a.
   */
  protected void phase3a() {
    add(new ConvolutionLayer(1, 1, 4096, 4096)
      .setPaddingXY(0, 0)
      .setAndFree(hdf5.readDataSet("param_0", "layer_34")
        .permuteDimensionsAndFree(fullyconnectedOrder))
    );
    add(new ImgBandBiasLayer(4096)
      .setAndFree((hdf5.readDataSet("param_1", "layer_34"))));
    add(new ActivationLayer(ActivationLayer.Mode.RELU));
  
    add(new ConvolutionLayer(1, 1, 4096, 1000)
      .setPaddingXY(0, 0)
      .setAndFree(hdf5.readDataSet("param_0", "layer_36")
        .permuteDimensionsAndFree(fullyconnectedOrder))
    );
    add(new ImgBandBiasLayer(1000)
      .setAndFree((hdf5.readDataSet("param_1", "layer_36"))));
  }
  
  /**
   * Add pooling layer.
   *
   * @param size the size
   */
  protected void addPoolingLayer(final int size) {
    if (large) {
      add(new ImgModulusPaddingLayer(size, size));
    }
    else {
      add(new ImgModulusPaddingLayer(-size, -size));
    }
    add(new PoolingLayer()
      .setMode(PoolingLayer.PoolingMode.Max)
      .setWindowXY(size, size)
      .setStrideXY(size, size));
  }
  
  /**
   * Add convolution layer.
   *
   * @param radius         the radius
   * @param inputBands     the input bands
   * @param outputBands    the output bands
   * @param activationMode the activation mode
   * @param hdf_group      the hdf group
   */
  protected void addConvolutionLayer(final int radius, final int inputBands, final int outputBands, final ActivationLayer.Mode activationMode, final String hdf_group) {
    add(new ConvolutionLayer(radius, radius, inputBands, outputBands)
      .setPaddingXY(0, 0)
      .setAndFree(hdf5.readDataSet("param_0", hdf_group)
        .permuteDimensionsAndFree(convolutionOrder))
    );
    add(new ImgBandBiasLayer(outputBands)
      .setAndFree((hdf5.readDataSet("param_1", hdf_group))));
    add(new ActivationLayer(activationMode));
  }
  
  /**
   * Phase 3 b.
   */
  protected void phase3b() {
    add(new BandReducerLayer()
      .setMode(getFinalPoolingMode()));
    add(new SoftmaxActivationLayer());
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
  
  /**
   * Is large boolean.
   *
   * @return the boolean
   */
  public boolean isLarge() {
    return large;
  }
  
  /**
   * Sets large.
   *
   * @param large the large
   * @return the large
   */
  public VGG16_HDF5 setLarge(boolean large) {
    this.large = large;
    return this;
  }
  
  /**
   * Is dense boolean.
   *
   * @return the boolean
   */
  public boolean isDense() {
    return dense;
  }
  
  /**
   * Sets dense.
   *
   * @param dense the dense
   * @return the dense
   */
  public VGG16_HDF5 setDense(boolean dense) {
    this.dense = dense;
    return this;
  }
  
  /**
   * Gets final pooling mode.
   *
   * @return the final pooling mode
   */
  public PoolingLayer.PoolingMode getFinalPoolingMode() {
    return finalPoolingMode;
  }
  
  /**
   * Sets final pooling mode.
   *
   * @param finalPoolingMode the final pooling mode
   * @return the final pooling mode
   */
  public VGG16_HDF5 setFinalPoolingMode(PoolingLayer.PoolingMode finalPoolingMode) {
    this.finalPoolingMode = finalPoolingMode;
    return this;
  }
  
  /**
   * The type Jblas.
   */
  public static class JBLAS extends VGG16_HDF5 {
  
    /**
     * The Samples.
     */
    int samples = 3;
  
    /**
     * Instantiates a new Vgg 16 hdf 5.
     *
     * @param hdf5 the hdf 5
     */
    public JBLAS(final Hdf5Archive hdf5) {
      super(hdf5);
      setLarge(true);
      fullyconnectedOrder = new int[]{0, 1};
    }
    
    @Override
    public Layer buildNetwork() {
      prototype = new Tensor(224, 224, 3);
      //  model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
      add(new AssertDimensionsLayer(224, 224, 3));
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(64, 3, 3, activation='relu'))
      add(new ConvolutionLayer(3, 3, 3, 64)
        .setPaddingXY(0, 0)
        .set(hdf5.readDataSet("param_0", "layer_1")
          .permuteDimensions(convolutionOrder)));
      add(new ImgBandBiasLayer(64)
        .set((hdf5.readDataSet("param_1", "layer_1"))));
      add(new ActivationLayer(ActivationLayer.Mode.RELU));
      
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(64, 3, 3, activation='relu'))
      addConvolution(3, 64, 64, "layer_3");
      //  model.add(MaxPooling2D((2,2), strides=(2,2)))
      addPoolingLayer(2);
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(128, 3, 3, activation='relu'))
      addConvolution(3, 64, 128, "layer_6");
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(128, 3, 3, activation='relu'))
      addConvolution(3, 128, 128, "layer_8");
      //  model.add(MaxPooling2D((2,2), strides=(2,2)))
      addPoolingLayer(2);
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(256, 3, 3, activation='relu'))
      addConvolution(3, 128, 256, "layer_11");
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(256, 3, 3, activation='relu'))
      addConvolution(3, 256, 256, "layer_13");
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(256, 3, 3, activation='relu'))
      addConvolution(3, 256, 256, "layer_15");
      //  model.add(MaxPooling2D((2,2), strides=(2,2)))
      addPoolingLayer(2);
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(512, 3, 3, activation='relu'))
      addConvolution(3, 256, 512, "layer_18");
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(512, 3, 3, activation='relu'))
      addConvolution(3, 512, 512, "layer_20");
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(512, 3, 3, activation='relu'))
      addConvolution(3, 512, 512, "layer_22");
      //  model.add(MaxPooling2D((2,2), strides=(2,2)))
      addPoolingLayer(2);
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(512, 3, 3, activation='relu'))
      addConvolution(3, 512, 512, "layer_25");
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(512, 3, 3, activation='relu'))
      addConvolution(3, 512, 512, "layer_27");
      //  model.add(ZeroPadding2D((1,1)))
      add(new ImgZeroPaddingLayer(1, 1));
      //  model.add(Convolution2D(512, 3, 3, activation='relu'))
      addConvolution(3, 512, 512, "layer_29");
      //  model.add(MaxPooling2D((2,2), strides=(2,2)))
      addPoolingLayer(2);
      //  model.add(Flatten())
      //  model.add(Dense(4096, activation='relu'))
      add(new com.simiacryptus.mindseye.layers.java.FullyConnectedLayer(new int[]{25088}, new int[]{4096})
        .set(hdf5.readDataSet("param_0", "layer_32")
          .permuteDimensions(fullyconnectedOrder))
        .setName("fullyconnected_32"));
      add(new BiasLayer(4096)
        .set((hdf5.readDataSet("param_1", "layer_32"))));
      //  model.add(Dropout(0.5))
      //model.add(new DropoutNoiseLayer(0.5));
      //  model.add(Dense(4096, activation='relu'))
      add(new com.simiacryptus.mindseye.layers.java.FullyConnectedLayer(new int[]{4096}, new int[]{4096})
        .set(hdf5.readDataSet("param_0", "layer_34")
          .permuteDimensions(fullyconnectedOrder))
      );
      add(new BiasLayer(4096)
        .set((hdf5.readDataSet("param_1", "layer_34"))));
      //  model.add(Dropout(0.5))
      //model.add(new DropoutNoiseLayer(0.5));
      //  model.add(Dense(1000, activation='softmax'))
      add(new com.simiacryptus.mindseye.layers.java.FullyConnectedLayer(new int[]{4096}, new int[]{1000})
        .set(hdf5.readDataSet("param_0", "layer_36")
          .permuteDimensions(fullyconnectedOrder))
        .setName("fullyconnected_36"));
      add(new BiasLayer(1000)
        .set((hdf5.readDataSet("param_1", "layer_36")))
        .setName("bias_36"));
      add(new SoftmaxActivationLayer());
      setPrecision(pipelineNetwork);
      return pipelineNetwork;
    }
    
    public void addPoolingLayer(final int size) {
      add(new PoolingLayer()
        .setMode(PoolingLayer.PoolingMode.Max)
        .setWindowXY(size, size)
        .setStrideXY(size, size));
    }
  
    /**
     * Add convolution.
     *
     * @param radius      the radius
     * @param inputBands  the input bands
     * @param outputBands the output bands
     * @param layer       the layer
     */
    public void addConvolution(final int radius, final int inputBands, final int outputBands, final String layer) {addConvolutionLayer(radius, inputBands, outputBands, ActivationLayer.Mode.RELU, layer);}
    
  }
  
  /**
   * The type Noisy.
   */
  public static class Noisy extends VGG16_HDF5 {
  
    private int samples;
    private double density;
  
    /**
     * Instantiates a new Vgg 16 hdf 5.
     *
     * @param hdf5 the hdf 5
     */
    public Noisy(final Hdf5Archive hdf5) {
      super(hdf5);
      density = 0.5;
      samples = 3;
    }
    
    protected void phase3a() {
      PipelineNetwork stochasticNet = new PipelineNetwork(1);
      
      DAGNode prev = stochasticNet.getHead();
      stochasticNet.wrap(new GateProductLayer(), prev,
        stochasticNet.add(new StochasticBinaryNoiseLayer(density, 1.0 / density, 1, 1, 4096), new DAGNode[]{}));
      
      stochasticNet.wrap(new ConvolutionLayer(1, 1, 4096, 4096)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_34")
          .permuteDimensionsAndFree(fullyconnectedOrder))
        .setPrecision(precision)
        .explode()
      );
      stochasticNet.wrap(new ImgBandBiasLayer(4096)
        .setAndFree((hdf5.readDataSet("param_1", "layer_34"))));
      
      prev = stochasticNet.getHead();
      stochasticNet.wrap(new GateProductLayer(), prev,
        stochasticNet.add(new StochasticBinaryNoiseLayer(density, 1.0 / density, 1, 1, 4096), new DAGNode[]{}));
      
      stochasticNet.wrap(new ActivationLayer(ActivationLayer.Mode.RELU));
      stochasticNet.wrap(new ConvolutionLayer(1, 1, 4096, 1000)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_36")
          .permuteDimensionsAndFree(fullyconnectedOrder))
        .setPrecision(precision)
        .explode()
      );
      stochasticNet.wrap(new ImgBandBiasLayer(1000)
        .setAndFree((hdf5.readDataSet("param_1", "layer_36"))));
      
      add(new StochasticSamplingSubnetLayer(stochasticNet, samples));
    }
  
    /**
     * The Samples.
     */
    public int getSamples() {
      return samples;
    }
  
    public Noisy setSamples(int samples) {
      this.samples = samples;
      return this;
    }
  
    public double getDensity() {
      return density;
    }
  
    public Noisy setDensity(double density) {
      this.density = density;
      return this;
    }
  }
  
}
