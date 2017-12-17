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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

/**
 * The type Convolution layer.
 */
@SuppressWarnings("serial")
public class ConvolutionLayer extends NNLayer implements LayerPrecision<ConvolutionLayer> {
  
  /**
   * The Filter.
   */
  public final Tensor kernel;
  /**
   * The Stride x.
   */
  int strideX = 1;
  /**
   * The Stride y.
   */
  int strideY = 1;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected ConvolutionLayer() {
    this((Tensor) null);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width  the width
   * @param height the height
   * @param bands  the bands
   */
  public ConvolutionLayer(final int width, final int height, final int bands) {
    this(new Tensor(width, height, bands));
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
    this(width, height, inputBands * outputBands);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json the json
   */
  protected ConvolutionLayer(final JsonObject json) {
    super(json);
    kernel = Tensor.fromJson(json.get("filter"));
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param kernel the filter
   */
  protected ConvolutionLayer(final Tensor kernel) {
    super();
    if (kernel.getDimensions().length != 3) throw new IllegalArgumentException();
    if (kernel.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if (kernel.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if (kernel.getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.kernel = kernel;
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
   * @return the convolution layer
   */
  public static ConvolutionLayer fromJson(final JsonObject json) {
    return new ConvolutionLayer(json);
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
      subLayers.add(new SimpleConvolutionLayer(batchKernel)
        .setStrideX(strideX).setStrideY(strideY).setPrecision(precision));
    }
    final DAGNode input = network.getHead();
    network.add(new ImgConcatLayer().setMaxBands(outputBands).setPrecision(precision),
      subLayers.stream().map(l -> {
        return network.add(l, input);
      }).toArray(i -> new DAGNode[i]));
    if (isFrozen()) {
      network.freeze();
    }
    final NNResult innerResult = network.eval(nncontext, inObj);
    return new NNResult(innerResult.getData()) {
      @Override
      public void accumulate(final DeltaSet<NNLayer> xxx, final TensorList data) {
        innerResult.accumulate(xxx, data);
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
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    json.add("filter", kernel.toJson());
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("precision", precision.name());
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
}
