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

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

/**
 * This is the general convolution layer, allowing any number of input and output bands. During execution it delegates
 * processing to a dynamically created subbnet created using SimpleConvolutionLayer and ImgConcatLayer to implement the
 * more general layer contract.
 */
@SuppressWarnings("serial")
public class ConvolutionLayer extends NNLayer implements LayerPrecision<ConvolutionLayer> {
  
  /**
   * The Filter.
   */
  public final Tensor kernel;
  private final int inputBands;
  private final int outputBands;
  private int strideX = 1;
  private int strideY = 1;
  private Integer paddingX = null;
  private Integer paddingY = null;
  private Precision precision = Precision.Double;
  
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
    this.kernel = new Tensor(width, height, inputBands * outputBands);
    if (kernel.getDimensions().length != 3) throw new IllegalArgumentException();
    if (kernel.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if (kernel.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if (kernel.getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.inputBands = inputBands;
    this.outputBands = outputBands;
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ConvolutionLayer(final JsonObject json, Map<String, byte[]> resources) {
    super(json);
    this.kernel = Tensor.fromJson(json.get("filter"), resources);
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
   * Add.
   *
   * @param f    the f
   * @param data the data
   */
  public static void add(final DoubleSupplier f, final double[] data) {
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
  public static ConvolutionLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ConvolutionLayer(json, rs);
  }
  
  /**
   * Add weights convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  public ConvolutionLayer addWeights(final DoubleSupplier f) {
    ConvolutionLayer.add(f, kernel.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    assert 1 == inObj.length;
    assert 3 == inObj[0].getData().getDimensions().length;
    assert inputBands == inObj[0].getData().getDimensions()[2];
    final PipelineNetwork network = new PipelineNetwork();
    final List<SimpleConvolutionLayer> subLayers = new ArrayList<>();
    // Extract Weights
    final int[] filterDimensions = kernel.getDimensions();
    final int[] inputDimensions = inObj[0].getData().getDimensions();
    final int inputBands = inputDimensions[2];
    final int outputBands = filterDimensions[2] / inputBands;
    final int inputBandsSq = inputBands * inputBands;
    for (int offset = 0; offset < filterDimensions[2]; offset += inputBandsSq) {
      final Tensor batchKernel = new Tensor(filterDimensions[0], filterDimensions[1], inputBandsSq);
      final int _offset = offset;
      batchKernel.setByCoord(batchCoord -> {
        final int filterBandT = getFilterBand(inputBands, outputBands, _offset, batchCoord);
        if (_offset + batchCoord.getCoords()[2] < filterDimensions[2]) {
          return kernel.get(batchCoord.getCoords()[0], batchCoord.getCoords()[1], filterBandT);
        }
        else {
          return 0;
        }
      });
      SimpleConvolutionLayer simpleConvolutionLayer = new SimpleConvolutionLayer(batchKernel)
        .setStrideX(getStrideX()).setStrideY(getStrideY()).setPrecision(precision);
      if (paddingX != null) simpleConvolutionLayer.setPaddingX(paddingX);
      if (paddingY != null) simpleConvolutionLayer.setPaddingY(paddingY);
      subLayers.add(simpleConvolutionLayer);
    }
    final DAGNode input = network.getHead();
    network.add(new ImgConcatLayer().setMaxBands(outputBands).setPrecision(precision),
                subLayers.stream().map(l -> {
                  return network.add(l, input);
                }).toArray(i -> new DAGNode[i]));
    if (isFrozen()) {
      network.freeze();
    }
    final NNResult result = network.eval(nncontext, inObj);
    assert 1 == inObj.length;
    assert inObj[0].getData().length() == result.getData().length();
    assert 3 == result.getData().getDimensions().length;
    assert outputBands == result.getData().getDimensions()[2];
    return new NNResult(result.getData()) {
  
      @Override
      public void finalize() {
        result.finalize();
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> xxx, final TensorList data) {
        result.accumulate(xxx, data);
        // Extract Deltas
        final Tensor filterDelta = new Tensor(filterDimensions);
        for (int batchNumber = 0; batchNumber < subLayers.size(); batchNumber++) {
          final SimpleConvolutionLayer batchLayer = subLayers.get(batchNumber);
          final Delta<NNLayer> subnetDelta = xxx.getMap().remove(batchLayer);
          if (null != subnetDelta) {
            final int[] batchDimensions = batchLayer.kernel.getDimensions();
            final Tensor batchDelta = new Tensor(null == subnetDelta ? null : subnetDelta.getDelta(), batchDimensions);
            final int offset = batchNumber * inputBandsSq;
            batchDelta.coordStream().forEach(batchCoord -> {
              if (offset + batchCoord.getCoords()[2] < filterDimensions[2]) {
                final int bandT = getFilterBand(inputBands, outputBands, offset, batchCoord);
                filterDelta.set(batchCoord.getCoords()[0], batchCoord.getCoords()[1], bandT, batchDelta.get(batchCoord));
              }
            });
          }
        }
        xxx.get(ConvolutionLayer.this, kernel.getData()).addInPlace(filterDelta.getData());
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
    };
  }
  
  /**
   * Gets filter band.
   *
   * @param inputBands  the input bands
   * @param outputBands the output bands
   * @param offset      the offset
   * @param coord       the coord
   * @return the filter band
   */
  public int getFilterBand(final int inputBands, final int outputBands, final int offset, final Coordinate coord) {
    final int filterBand = offset + coord.getCoords()[2];
    final int filterBandX = filterBand % inputBands;
    final int filterBandY = (filterBand - filterBandX) / inputBands;
    assert filterBand == filterBandY * inputBands + filterBandX;
    return filterBandX * outputBands + filterBandY;
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("filter", kernel.toJson(resources, dataSerializer));
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
  public ConvolutionLayer set(final DoubleSupplier f) {
    return set(i -> f.getAsDouble());
  }
  
  /**
   * Set convolution layer.
   *
   * @param tensor the tensor
   * @return the convolution layer
   */
  public ConvolutionLayer set(final Tensor tensor) {
    kernel.set(tensor);
    return this;
  }
  
  /**
   * Set convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  public ConvolutionLayer set(final IntToDoubleFunction f) {
    kernel.set(f);
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(kernel.getData());
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
  public ConvolutionLayer setPaddingXY(Integer x, Integer y) {
    return setPaddingX(x).setPaddingY(y);
  }
  
  /**
   * Gets padding x.
   *
   * @return the padding x
   */
  public Integer getPaddingX() {
    return paddingX;
  }
  
  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   * @return the padding x
   */
  public ConvolutionLayer setPaddingX(Integer paddingX) {
    this.paddingX = paddingX;
    return this;
  }
  
  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  public Integer getPaddingY() {
    return paddingY;
  }
  
  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   * @return the padding y
   */
  public ConvolutionLayer setPaddingY(Integer paddingY) {
    this.paddingY = paddingY;
    return this;
  }
}
