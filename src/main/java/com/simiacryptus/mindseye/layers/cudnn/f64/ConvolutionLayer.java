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

package com.simiacryptus.mindseye.layers.cudnn.f64;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.util.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;

/**
 * The type Convolution layer.
 */
public class ConvolutionLayer extends NNLayer {
  
  
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
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json the json
   */
  protected ConvolutionLayer(JsonObject json) {
    super(json);
    this.kernel = Tensor.fromJson(json.getAsJsonObject("filter"));
    this.strideX = json.get("strideX").getAsInt();
    this.strideY = json.get("strideY").getAsInt();
  }
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected ConvolutionLayer() {
    this((Tensor) null);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param kernel the filter
   */
  protected ConvolutionLayer(Tensor kernel) {
    super();
    if (kernel.getDimensions().length != 3) throw new IllegalArgumentException();
    if (kernel.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if (kernel.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if (kernel.getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.kernel = kernel;
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width  the width
   * @param height the height
   * @param bands  the bands
   */
  public ConvolutionLayer(final int width, int height, final int bands) {
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
  public ConvolutionLayer(final int width, int height, final int inputBands, final int outputBands) {
    this(width, height, inputBands * outputBands);
  }
  
  /**
   * From json convolution layer.
   *
   * @param json the json
   * @return the convolution layer
   */
  public static ConvolutionLayer fromJson(JsonObject json) {
    return new ConvolutionLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("filter", kernel.getJson());
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    return json;
  }
  
  /**
   * Add weights convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  public ConvolutionLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.kernel.getData());
    return this;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    PipelineNetwork network = new PipelineNetwork();
    List<SimpleConvolutionLayer> subLayers = new ArrayList<>();
    // Extract Weights
    int[] filterDimensions = kernel.getDimensions();
    int[] inputDimensions = inObj[0].getData().getDimensions();
    int inputBands = inputDimensions[2];
    int outputBands = filterDimensions[2] / inputBands;
    int inputBandsSq = inputBands * inputBands;
    for (int offset = 0; offset < filterDimensions[2]; offset += inputBandsSq) {
      Tensor batchKernel = new Tensor(filterDimensions[0], filterDimensions[1], inputBandsSq);
      int _offset = offset;
      batchKernel.fillByCoord(batchCoord -> {
        int filterBandT = getFilterBand(inputBands, outputBands, _offset, batchCoord);
        if (_offset + batchCoord.coords[2] < filterDimensions[2]) {
          return kernel.get(batchCoord.coords[0], batchCoord.coords[1], filterBandT);
        }
        else {
          return 0;
        }
      });
      subLayers.add(new SimpleConvolutionLayer(batchKernel)
        .setStrideX(strideX).setStrideY(strideY));
    }
    
    DAGNode input = network.getHead();
    network.add(new ImgConcatLayer().setMaxBands(outputBands),
      subLayers.stream().map(l -> {
        return network.add(l, input);
      }).toArray(i -> new DAGNode[i]));
    if (isFrozen()) network.freeze();
    NNResult innerResult = network.eval(nncontext, inObj);
    return new NNResult(innerResult.getData()) {
      @Override
      public void accumulate(DeltaSet<NNLayer> deltaSet, TensorList data) {
        innerResult.accumulate(deltaSet, data);
        // Extract Deltas
        Tensor filterDelta = new Tensor(filterDimensions);
        for (int batchNumber = 0; batchNumber < subLayers.size(); batchNumber++) {
          SimpleConvolutionLayer batchLayer = subLayers.get(batchNumber);
          Delta<NNLayer> subnetDelta = deltaSet.getMap().remove(batchLayer);
          if (null != subnetDelta) {
            int[] batchDimensions = batchLayer.kernel.getDimensions();
            Tensor batchDelta = new Tensor(null == subnetDelta ? null : subnetDelta.getDelta(), batchDimensions);
            int offset = batchNumber * inputBandsSq;
            batchDelta.coordStream().forEach(batchCoord -> {
              if (offset + batchCoord.coords[2] < filterDimensions[2]) {
                int bandT = getFilterBand(inputBands, outputBands, offset, batchCoord);
                filterDelta.set(batchCoord.coords[0], batchCoord.coords[1], bandT, batchDelta.get(batchCoord));
              }
            });
          }
        }
        deltaSet.get(ConvolutionLayer.this, ConvolutionLayer.this.kernel.getData()).addInPlace(filterDelta.getData());
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
  public int getFilterBand(int inputBands, int outputBands, int offset, Coordinate coord) {
    int filterBand = offset + coord.coords[2];
    int filterBandX = filterBand % inputBands;
    int filterBandY = (filterBand - filterBandX) / inputBands;
    assert filterBand == filterBandY * inputBands + filterBandX;
    return filterBandX * outputBands + filterBandY;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public ConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    this.kernel.coordStream().parallel().forEach(c -> {
      this.kernel.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public ConvolutionLayer setWeights(final DoubleSupplier f) {
    this.kernel.coordStream().parallel().forEach(c -> {
      this.kernel.set(c, f.getAsDouble());
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.kernel.getData());
  }
  
  
  /**
   * Sets weights log.
   *
   * @param w the w
   * @return the weights log
   */
  public ConvolutionLayer setWeightsLog(double w) {
    setWeights(() -> Math.exp(w) * (Math.random() - 0.5));
    return this;
  }
  
  /**
   * Sets stride xy.
   *
   * @param x the x
   * @param y the y
   * @return the stride xy
   */
  public ConvolutionLayer setStrideXY(int x, int y) {
    strideX = x;
    strideY = y;
    return this;
  }
  
}
