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
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgMinSizeLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgModulusPaddingLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;
import com.simiacryptus.mindseye.layers.cudnn.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.StochasticSamplingSubnetLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.ImgReshapeLayer;
import com.simiacryptus.mindseye.layers.java.StochasticBinaryNoiseLayer;
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
public class VGG19_HDF5 extends VGG16 implements NetworkFactory, HasHDF5 {
  
  /**
   * The constant log.
   */
  protected static final Logger log = LoggerFactory.getLogger(VGG19_HDF5.class);
  /**
   * The Pipeline network.
   */
  protected final PipelineNetwork pipeline = new PipelineNetwork();
  /**
   * The Hdf 5.
   */
  protected final Hdf5Archive hdf5;
  /**
   * The Convolution order.
   */
  @Nonnull
  int[] convolutionOrder = {3, 2, 0, 1};
  /**
   * The Fullyconnected order.
   */
  @Nonnull
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
  public VGG19_HDF5(Hdf5Archive hdf5) {this.hdf5 = hdf5;}
  
  /**
   * Add.
   *
   * @param layer the layer
   */
  protected void add(@Nonnull Layer layer) {
    Tensor newValue = evaluatePrototype(add(layer, pipeline), this.prototype, cnt++);
    if (null != this.prototype) this.prototype.freeRef();
    this.prototype = newValue;
  }
  
  public Layer buildNetwork() {
    try {
      if (null != this.prototype) this.prototype.freeRef();
      prototype = new Tensor(226, 226, 3);
      phase0();
      phase1();
      phase2();
      phase3();
      return pipeline;
    } finally {
      prototype.freeRef();
      prototype = null;
    }
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
   * Phase 0.
   */
  protected void phase0() {
    add(new ImgMinSizeLayer(226, 226));
    Tensor tensor = new Tensor(-103.939, -116.779, -123.68);
    add(new ImgBandBiasLayer(3).setAndFree(tensor));
  }
  
  /**
   * Phase 1 a.
   */
  protected void phase1a() {
    addConvolutionLayer(3, 3, 64, ActivationLayer.Mode.RELU, "layer_1");
    addConvolutionLayer(3, 64, 64, ActivationLayer.Mode.RELU, "layer_3");
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
   * Phase 1 c.
   */
  protected void phase1c() {
    addPoolingLayer(2);
    addConvolutionLayer(3, 128, 256, ActivationLayer.Mode.RELU, "layer_11");
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_13");
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_15");
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_17");
  }
  
  /**
   * Phase 1 d.
   */
  protected void phase1d() {
    addPoolingLayer(2);
    addConvolutionLayer(3, 256, 512, ActivationLayer.Mode.RELU, "layer_20");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_22");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_24");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_26");
  }
  
  /**
   * Phase 1 e.
   */
  protected void phase1e() {
    addPoolingLayer(2);
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_29");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_31");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_33");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_35");
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
        .setAndFree(hdf5.readDataSet("param_0", "layer_38")
          .reshapeCastAndFree(7, 7, 512, 4096).permuteDimensionsAndFree(0, 1, 3, 2)
        )
      );
    }
    else {
      add(new ImgModulusPaddingLayer(7, 7));
      add(new ImgReshapeLayer(7, 7, false));
      add(new ConvolutionLayer(1, 1, 25088, 4096)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_38")
          .permuteDimensionsAndFree(fullyconnectedOrder))
      );
    }
    
    add(new ImgBandBiasLayer(4096)
      .setAndFree((hdf5.readDataSet("param_1", "layer_38"))));
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
      .setAndFree(hdf5.readDataSet("param_0", "layer_40")
        .permuteDimensionsAndFree(fullyconnectedOrder))
    );
    add(new ImgBandBiasLayer(4096)
      .setAndFree((hdf5.readDataSet("param_1", "layer_40"))));
    add(new ActivationLayer(ActivationLayer.Mode.RELU));
    
    add(new ConvolutionLayer(1, 1, 4096, 1000)
      .setPaddingXY(0, 0)
      .setAndFree(hdf5.readDataSet("param_0", "layer_42")
        .permuteDimensionsAndFree(fullyconnectedOrder))
    );
    add(new ImgBandBiasLayer(1000)
      .setAndFree((hdf5.readDataSet("param_1", "layer_42"))));
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
    add(new SoftmaxActivationLayer()
      .setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE)
      .setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL));
    add(new BandReducerLayer()
      .setMode(getFinalPoolingMode()));
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
  public ImageClassifier setLarge(boolean large) {
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
  public ImageClassifier setDense(boolean dense) {
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
  public ImageClassifier setFinalPoolingMode(PoolingLayer.PoolingMode finalPoolingMode) {
    this.finalPoolingMode = finalPoolingMode;
    return this;
  }
  
  
  /**
   * The type Noisy.
   */
  public static class Noisy extends VGG19_HDF5 {
    
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
      stochasticNet.wrap(new ProductLayer(), prev,
        stochasticNet.add(new StochasticBinaryNoiseLayer(density, 1.0 / density, 1, 1, 4096), new DAGNode[]{})).freeRef();
      
      stochasticNet.wrap(new ConvolutionLayer(1, 1, 4096, 4096)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_40")
          .permuteDimensionsAndFree(fullyconnectedOrder))
        .setPrecision(precision)
        .explode()
      ).freeRef();
      stochasticNet.wrap(new ImgBandBiasLayer(4096)
        .setAndFree((hdf5.readDataSet("param_1", "layer_40")))).freeRef();
      
      prev = stochasticNet.getHead();
      stochasticNet.wrap(new ProductLayer(), prev,
        stochasticNet.add(new StochasticBinaryNoiseLayer(density, 1.0 / density, 1, 1, 4096), new DAGNode[]{})).freeRef();
  
      stochasticNet.wrap(new ActivationLayer(ActivationLayer.Mode.RELU)).freeRef();
      stochasticNet.wrap(new ConvolutionLayer(1, 1, 4096, 1000)
        .setPaddingXY(0, 0)
        .setAndFree(hdf5.readDataSet("param_0", "layer_42")
          .permuteDimensionsAndFree(fullyconnectedOrder))
        .setPrecision(precision)
        .explode()
      ).freeRef();
      stochasticNet.wrap(new ImgBandBiasLayer(1000)
        .setAndFree((hdf5.readDataSet("param_1", "layer_42")))).freeRef();
      
      add(new StochasticSamplingSubnetLayer(stochasticNet, samples));
    }
  
    /**
     * The Samples.
     *
     * @return the samples
     */
    public int getSamples() {
      return samples;
    }
  
    /**
     * Sets samples.
     *
     * @param samples the samples
     * @return the samples
     */
    public Noisy setSamples(int samples) {
      this.samples = samples;
      return this;
    }
  
    /**
     * Gets density.
     *
     * @return the density
     */
    public double getDensity() {
      return density;
    }
  
    /**
     * Sets density.
     *
     * @param density the density
     * @return the density
     */
    public Noisy setDensity(double density) {
      this.density = density;
      return this;
    }
  }
  
}
