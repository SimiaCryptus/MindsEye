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
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.util.Arrays;

/**
 * Details about this network architecture can be found in the following arXiv paper: Very Deep Convolutional Networks
 * for Large-Scale Image Recognition K. Simonyan, A. Zisserman arXiv:1409.1556 Please cite the paper if you use the
 * models.
 */
public class VGG16_HDF5 extends VGG16 implements NetworkFactory, HasHDF5 {
  /**
   * The constant log.
   */
  protected static final Logger log = LoggerFactory.getLogger(Hdf5Archive.class);
  /**
   * The Hdf 5.
   */
  protected final Hdf5Archive hdf5;
  /**
   * The Network.
   */
  protected volatile Layer network;
  /**
   * The Model.
   */
  @javax.annotation.Nonnull
  protected
  PipelineNetwork model = new PipelineNetwork();
  /**
   * The Prototype.
   */
  @Nullable
  Tensor prototype = new Tensor(224, 224, 3);
  /**
   * The Cnt.
   */
  int cnt = 1;
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
  /**
   * The Precision.
   */
  @javax.annotation.Nonnull
  Precision precision = Precision.Float;
  private boolean large = true;
  private boolean dense = true;
  private PoolingLayer.PoolingMode finalPoolingMode = PoolingLayer.PoolingMode.Avg;
  private int stochasticSamples = 1;
  
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
   * @param model the model
   * @return the layer
   */
  protected static Layer add(Layer layer, PipelineNetwork model) {
    name(layer);
    if (layer instanceof Explodable) {
      Layer explode = ((Explodable) layer).explode();
      try {
        if (explode instanceof DAGNetwork) {
          ((DAGNetwork) explode).visitNodes(node -> name(node.getLayer()));
          log.info(String.format("Exploded %s to %s (%s nodes)", layer.getName(), explode.getClass().getSimpleName(), ((DAGNetwork) explode).getNodes().size()));
        }
        else {
          log.info(String.format("Exploded %s to %s (%s nodes)", layer.getName(), explode.getClass().getSimpleName(), explode.getName()));
        }
        return add(explode, model);
      } finally {
        layer.freeRef();
      }
    }
    else {
      model.wrap(layer);
      return layer;
    }
  }
  
  /**
   * Evaluate prototype tensor.
   *
   * @param layer         the layer
   * @param prevPrototype the prev prototype
   * @param cnt           the cnt
   * @return the tensor
   */
  protected static Tensor evaluatePrototype(final Layer layer, final Tensor prevPrototype, int cnt) {
    int numberOfParameters = layer.state().stream().mapToInt(x -> x.length).sum();
    @javax.annotation.Nonnull int[] prev_dimensions = prevPrototype.getDimensions();
    Result eval = layer.eval(prevPrototype);
    TensorList newPrototype = eval.getData();
    if (null != prevPrototype) prevPrototype.freeRef();
    eval.freeRef();
    try {
      @javax.annotation.Nonnull int[] new_dimensions = newPrototype.getDimensions();
      log.info(String.format("Added layer #%d: %s; %s params, dimensions %s (%s) -> %s (%s)", //
        cnt, layer, numberOfParameters, //
        Arrays.toString(prev_dimensions), Tensor.length(prev_dimensions), //
        Arrays.toString(new_dimensions), Tensor.length(new_dimensions)));
      return newPrototype.get(0);
    } finally {
      newPrototype.freeRef();
    }
  }
  
  /**
   * Name.
   *
   * @param layer the layer
   */
  protected static void name(final Layer layer) {
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
  public Layer build() {
    try {
      phase0();
      phase1(large);
      phase2(dense, large);
      phase3();
      setPrecision();
      if (null != prototype) prototype.freeRef();
      prototype = null;
      return model;
    } catch (@javax.annotation.Nonnull final RuntimeException e) {
      throw e;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Sets precision.
   *
   */
  protected void setPrecision() {
    model.visitLayers(layer -> {
      if (layer instanceof MultiPrecision) {
        ((MultiPrecision) layer).setPrecision(precision);
      }
    });
  }
  
  /**
   * Phase 0.
   *
   */
  protected void phase0() {
    add(new ImgMinSizeLayer(226, 226));
  }
  
  /**
   * Phase 1.
   *
   * @param large  the large
   */
  protected void phase1(final boolean large) {
    phase1a();
    phase1b();
    phase1c(large);
    phase1d(large);
    phase1e(large);
  }
  
  /**
   * Phase 1 a.
   *
   */
  protected void phase1a() {
    addConvolutionLayer(3, 3, 64, ActivationLayer.Mode.RELU, "layer_1");
    addConvolutionLayer(3, 64, 64, ActivationLayer.Mode.RELU, "layer_3");
  }
  
  /**
   * Phase 1 b.
   *
   */
  protected void phase1b() {
    addPoolingLayer(true);
    addConvolutionLayer(3, 64, 128, ActivationLayer.Mode.RELU, "layer_6");
    addConvolutionLayer(3, 128, 128, ActivationLayer.Mode.RELU, "layer_8");
  }
  
  /**
   * Phase 1 c.
   *
   * @param large  the large
   */
  protected void phase1c(final boolean large) {
    addPoolingLayer(large);
    addConvolutionLayer(3, 128, 256, ActivationLayer.Mode.RELU, "layer_11");
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_13");
    addConvolutionLayer(3, 256, 256, ActivationLayer.Mode.RELU, "layer_15");
  }
  
  /**
   * Phase 1 d.
   *
   * @param large  the large
   */
  protected void phase1d(final boolean large) {
    addPoolingLayer(large);
    addConvolutionLayer(3, 256, 512, ActivationLayer.Mode.RELU, "layer_18");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_20");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_22");
  }
  
  /**
   * Phase 1 e.
   *
   * @param large  the large
   */
  protected void phase1e(final boolean large) {
    addPoolingLayer(large);
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_25");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_27");
    addConvolutionLayer(3, 512, 512, ActivationLayer.Mode.RELU, "layer_29");
  }
  
  /**
   * Phase 2.
   *
   * @param dense the dense
   * @param large the large
   */
  protected void phase2(boolean dense, final boolean large) {
    phase2a();
    phase2b(dense, large);
  }
  
  /**
   * Phase 2 a.
   */
  protected void phase2a() {
    //  model.add(MaxPooling2D((2,2), strides=(2,2)))
    addPoolingLayer(true);
  }
  
  /**
   * Phase 2 b.
   *
   * @param dense the dense
   * @param large the large
   */
  protected void phase2b(final boolean dense, final boolean large) {
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
    if (1 == getStochasticSamples()) phase3a();
    else phase3a_sampled(getStochasticSamples());
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
  
  protected void phase3a_sampled(final int samples) {
    PipelineNetwork stochasticNet = new PipelineNetwork(1);
    
    DAGNode prev = stochasticNet.getHead();
    stochasticNet.wrap(new GateProductLayer(), prev,
      stochasticNet.add(new StochasticBinaryNoiseLayer(1.0, 1.0, 1, 1, 4096), new DAGNode[]{}));
    
    stochasticNet.wrap(new ConvolutionLayer(1, 1, 4096, 4096)
      .setPaddingXY(0, 0)
      .setAndFree(hdf5.readDataSet("param_0", "layer_34")
        .permuteDimensionsAndFree(fullyconnectedOrder))
    );
    stochasticNet.wrap(new ImgBandBiasLayer(4096)
      .setAndFree((hdf5.readDataSet("param_1", "layer_34"))));
    
    prev = stochasticNet.getHead();
    stochasticNet.wrap(new GateProductLayer(), prev,
      stochasticNet.add(new StochasticBinaryNoiseLayer(1.0, 1.0, 1, 1, 4096), new DAGNode[]{}));
    
    stochasticNet.wrap(new ActivationLayer(ActivationLayer.Mode.RELU));
    stochasticNet.wrap(new ConvolutionLayer(1, 1, 4096, 1000)
      .setPaddingXY(0, 0)
      .setAndFree(hdf5.readDataSet("param_0", "layer_36")
        .permuteDimensionsAndFree(fullyconnectedOrder))
    );
    stochasticNet.wrap(new ImgBandBiasLayer(1000)
      .setAndFree((hdf5.readDataSet("param_1", "layer_36"))));
    
    add(new StochasticSamplingSubnetLayer(stochasticNet, samples));
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
   * Add pooling layer.
   *
   * @param large the large
   */
  protected void addPoolingLayer(final boolean large) {
    if (large) {
      add(new ImgModulusPaddingLayer(2, 2));
    }
    else {
      add(new ImgModulusPaddingLayer(-2, -2));
    }
    add(new PoolingLayer()
      .setMode(PoolingLayer.PoolingMode.Max)
      .setWindowXY(2, 2)
      .setStrideXY(2, 2));
  }
  
  /**
   * Add convolution layer.
   *  @param radius         the radius
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
   * Add.
   *
   * @param layer the layer
   */
  protected void add(Layer layer) {
    this.prototype = evaluatePrototype(add(layer, model), this.prototype, cnt++);
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
  
  public PoolingLayer.PoolingMode getFinalPoolingMode() {
    return finalPoolingMode;
  }
  
  public VGG16_HDF5 setFinalPoolingMode(PoolingLayer.PoolingMode finalPoolingMode) {
    this.finalPoolingMode = finalPoolingMode;
    return this;
  }
  
  public int getStochasticSamples() {
    return stochasticSamples;
  }
  
  public VGG16_HDF5 setStochasticSamples(int stochasticSamples) {
    this.stochasticSamples = stochasticSamples;
    return this;
  }
}
