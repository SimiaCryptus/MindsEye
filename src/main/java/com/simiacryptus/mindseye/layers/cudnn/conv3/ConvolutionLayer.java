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

package com.simiacryptus.mindseye.layers.cudnn.conv3;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

/**
 * This is the general convolution layer, allowing any number of input and output bands at high scale. It implements an
 * explosion operation to produce a convolution network whose components have a managabe size and the same overall
 * function.
 */
@SuppressWarnings("serial")
public class ConvolutionLayer extends LayerBase implements MultiPrecision<ConvolutionLayer>, Explodable {
  
  @Nullable
  private final Tensor kernel;
  private final int inputBands;
  private final int outputBands;
  private int strideX = 1;
  private int strideY = 1;
  @Nullable
  private Integer paddingX = null;
  @Nullable
  private Integer paddingY = null;
  private Precision precision = Precision.Double;
  private int batchBands = 0;
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected ConvolutionLayer() {
    this(1, 1, 1, 1);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width       the width
   * @param height      the height
   * @param inputBands  the input bands
   * @param outputBands the output bands
   */
  public ConvolutionLayer(final int width, final int height, final int inputBands, final int outputBands) {
    super();
    assert 0 < width;
    assert 0 < height;
    assert 0 < inputBands;
    assert 0 < outputBands;
    this.kernel = new Tensor(width, height, inputBands * outputBands);
    if (getKernel().getDimensions().length != 3) throw new IllegalArgumentException();
    if (getKernel().getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if (getKernel().getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if (getKernel().getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    int batchBands = (int) Math.sqrt(CudaSettings.INSTANCE.getMaxFilterElements() / (width * height));
    //batchBands = binaryFriendly(batchBands, 3);
    setBatchBands(batchBands);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ConvolutionLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    this.kernel = Tensor.fromJson(json.get("filter"), resources);
    assert getKernel().isValid();
    this.setBatchBands(json.get("batchBands").getAsInt());
    this.setStrideX(json.get("strideX").getAsInt());
    this.setStrideY(json.get("strideY").getAsInt());
    JsonElement paddingX = json.get("paddingX");
    if (null != paddingX && paddingX.isJsonPrimitive()) this.setPaddingX((paddingX.getAsInt()));
    JsonElement paddingY = json.get("paddingY");
    if (null != paddingY && paddingY.isJsonPrimitive()) this.setPaddingY((paddingY.getAsInt()));
    this.precision = Precision.valueOf(json.get("precision").getAsString());
    this.inputBands = json.get("inputBands").getAsInt();
    this.outputBands = json.get("outputBands").getAsInt();
  }
  
  /**
   * Binary friendly int.
   *
   * @param value the value
   * @param bits  the bits
   * @return the int
   */
  public static int binaryFriendly(final int value, final int bits) {
    return (int) Math.pow(2, (Math.floor(Math.log(value) * bits) / bits) / Math.log(2));
  }
  
  /**
   * Add.
   *
   * @param f    the f
   * @param data the data
   */
  public static void add(@Nonnull final DoubleSupplier f, @Nonnull final double[] data) {
    for (int i = 0; i < data.length; i++) {
      data[i] += f.getAsDouble();
    }
  }
  
  /**
   * From json convolution layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the convolution layer
   */
  public static ConvolutionLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ConvolutionLayer(json, rs);
  }
  
  @Override
  protected void _free() {
    kernel.freeRef();
    super._free();
  }
  
  /**
   * Add weights convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  @Nonnull
  public ConvolutionLayer addWeights(@Nonnull final DoubleSupplier f) {
    ConvolutionLayer.add(f, getKernel().getData());
    return this;
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer.class);
  }
  
  /**
   * Explode nn layer.
   *
   * @return the nn layer
   */
  @Nonnull
  @Override
  public Layer explode() {
    @Nonnull ExplodedConvolutionGrid explodedNetwork = getExplodedNetwork();
    try {
      @Nonnull Layer network = explodedNetwork.getNetwork();
      network.setName(getName());
      return network;
    } finally {
      explodedNetwork.freeRef();
    }
  }
  
  /**
   * Gets exploded network.
   *
   * @return the exploded network
   */
  @Nonnull
  public ExplodedConvolutionGrid getExplodedNetwork() {
    assertAlive();
    int batchBands = getBatchBands();
    if (0 == batchBands) {
      batchBands = Math.max(inputBands, outputBands);
    }
//    if (batchBands > outputBands * 2) {
//      batchBands = outputBands;
//    }
    return new ExplodedConvolutionGrid(getConvolutionParams(), batchBands).write(kernel);
  }
  
  /**
   * Gets convolution params.
   *
   * @return the convolution params
   */
  @Nonnull
  public ConvolutionParams getConvolutionParams() {
    return new ConvolutionParams(inputBands, outputBands, precision, strideX, strideY, paddingX, paddingY, kernel.getDimensions());
  }
  
  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    final Tensor kernel = getKernel();
    kernel.addRef();
    assert kernel.isValid();
    assert 1 == inObj.length;
    assert 3 == inObj[0].getData().getDimensions().length;
    assert inputBands == inObj[0].getData().getDimensions()[2] : Arrays.toString(inObj[0].getData().getDimensions()) + "[2] != " + inputBands;
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    @Nonnull ExplodedConvolutionGrid grid = getExplodedNetwork();
    @Nonnull PipelineNetwork network = grid.getNetwork();
    if (isFrozen()) {
      network.freeze();
    }
    final Result result = network.evalAndFree(inObj);
    network.freeRef();
    final TensorList resultData = result.getData();
    assert inObj[0].getData().length() == resultData.length();
    assert 3 == resultData.getDimensions().length;
    assert outputBands == resultData.getDimensions()[2];
    ConvolutionLayer.this.addRef();
    return new Result(resultData, (@Nonnull final DeltaSet<Layer> deltaSet, @Nonnull final TensorList delta) -> {
      result.accumulate(deltaSet, delta);
      if (!isFrozen()) {
        Tensor read = grid.read(deltaSet, true);
        deltaSet.get(ConvolutionLayer.this, kernel.getData()).addInPlace(read.getData()).freeRef();
        read.freeRef();
      }
    }) {
      
      @Override
      public void accumulate(final DeltaSet<Layer> buffer, final TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }
      
      @Override
      protected void _free() {
        grid.freeRef();
        result.freeRef();
        kernel.freeRef();
        ConvolutionLayer.this.freeRef();
      }
      
      @Override
      public boolean isAlive() {
        return result.isAlive();
      }
    };
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("filter", getKernel().toJson(resources, dataSerializer));
    json.addProperty("batchBands", getBatchBands());
    json.addProperty("strideX", getStrideX());
    json.addProperty("strideY", getStrideY());
    json.addProperty("paddingX", getPaddingX());
    json.addProperty("paddingY", getPaddingY());
    json.addProperty("precision", precision.name());
    json.addProperty("inputBands", inputBands);
    json.addProperty("outputBands", outputBands);
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public ConvolutionLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @Nonnull
  public ConvolutionLayer set(@Nonnull final DoubleSupplier f) {
    return set(i -> f.getAsDouble());
  }
  
  /**
   * Set convolution layer.
   *
   * @param tensor the tensor
   * @return the convolution layer
   */
  @Nonnull
  public ConvolutionLayer set(@Nonnull final Tensor tensor) {
    getKernel().set(tensor);
    return this;
  }
  
  /**
   * Sets and free.
   *
   * @param tensor the tensor
   * @return the and free
   */
  @Nonnull
  public ConvolutionLayer setAndFree(@Nonnull final Tensor tensor) {
    set(tensor);
    tensor.freeRef();
    return this;
  }
  
  /**
   * Set convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  @Nonnull
  public ConvolutionLayer set(@Nonnull final IntToDoubleFunction f) {
    getKernel().set(f);
    return this;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(getKernel().getData());
  }
  
  /**
   * The Stride x.
   *
   * @return the stride x
   */
  public int getStrideX() {
    return strideX;
  }
  
  /**
   * Sets stride x.
   *
   * @param strideX the stride x
   * @return the stride x
   */
  @Nonnull
  public ConvolutionLayer setStrideX(int strideX) {
    this.strideX = strideX;
    return this;
  }
  
  /**
   * The Stride y.
   *
   * @return the stride y
   */
  public int getStrideY() {
    return strideY;
  }
  
  /**
   * Sets stride y.
   *
   * @param strideY the stride y
   * @return the stride y
   */
  @Nonnull
  public ConvolutionLayer setStrideY(int strideY) {
    this.strideY = strideY;
    return this;
  }
  
  /**
   * Sets weights log.
   *
   * @param f the f
   * @return the weights log
   */
  @Nonnull
  public ConvolutionLayer setWeightsLog(double f) {
    return set(() -> Math.pow(10, f) * (Math.random() - 0.5));
  }
  
  /**
   * Sets stride xy.
   *
   * @param x the x
   * @param y the y
   * @return the stride xy
   */
  @Nonnull
  public ConvolutionLayer setStrideXY(int x, int y) {
    return setStrideX(x).setStrideY(y);
  }
  
  /**
   * Sets padding xy.
   *
   * @param x the x
   * @param y the y
   * @return the padding xy
   */
  @Nonnull
  public ConvolutionLayer setPaddingXY(Integer x, Integer y) {
    return setPaddingX(x).setPaddingY(y);
  }
  
  /**
   * Gets padding x.
   *
   * @return the padding x
   */
  @Nullable
  public Integer getPaddingX() {
    return paddingX;
  }
  
  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   * @return the padding x
   */
  @Nonnull
  public ConvolutionLayer setPaddingX(Integer paddingX) {
    this.paddingX = paddingX;
    return this;
  }
  
  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  @Nullable
  public Integer getPaddingY() {
    return paddingY;
  }
  
  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   * @return the padding y
   */
  @Nonnull
  public ConvolutionLayer setPaddingY(Integer paddingY) {
    this.paddingY = paddingY;
    return this;
  }
  
  /**
   * The Filter.
   *
   * @return the kernel
   */
  @Nullable
  public Tensor getKernel() {
    return kernel;
  }
  
  
  /**
   * Gets batch bands.
   *
   * @return the batch bands
   */
  public int getBatchBands() {
    return batchBands;
  }
  
  /**
   * Sets batch bands.
   *
   * @param batchBands the batch bands
   * @return the batch bands
   */
  @Nonnull
  public ConvolutionLayer setBatchBands(int batchBands) {
    this.batchBands = batchBands;
    return this;
  }
}
