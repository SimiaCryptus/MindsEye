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

package com.simiacryptus.mindseye.layers.cudnn.conv;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.layers.cudnn.ImgLinearSubnetLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgZeroPaddingLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The higher level of convolution construction logic. Provides support for large numbers of input bands by splitting
 * the network into sub-networks that consider only a subset of the input bands, then summing the results together. This
 * strategy remains valid so long as the sub-networks are purely linear.
 */
class ExplodedConvolutionGrid extends ReferenceCountingBase {
  private static final Logger log = LoggerFactory.getLogger(ExplodedConvolutionGrid.class);
  
  /**
   * The Sub layers.
   */
  public final List<ExplodedConvolutionLeg> subLayers;
  /**
   * The Convolution params.
   */
  @Nonnull
  public final ConvolutionParams convolutionParams;
  
  /**
   * Instantiates a new Exploded network.
   *
   * @param convolutionParams the convolution params
   * @param maxBandBatch      the max band batch
   */
  public ExplodedConvolutionGrid(@Nonnull ConvolutionParams convolutionParams, int maxBandBatch) {
    this.convolutionParams = convolutionParams;
    int bandWidth = (maxBandBatch == 0) ? convolutionParams.inputBands : maxBandBatch;
    int rows = (int) Math.ceil((double) convolutionParams.inputBands / bandWidth);
    subLayers = IntStream.range(0, rows).map(x -> x * bandWidth).mapToObj(fromBand -> {
      int toBand = Math.min(convolutionParams.inputBands, fromBand + bandWidth);
      if (fromBand >= toBand) throw new RuntimeException(fromBand + " >= " + toBand);
      return new ExplodedConvolutionLeg(convolutionParams, fromBand, toBand);
    }).collect(Collectors.toList());
  }
  
  @Override
  protected void _free() {
    subLayers.stream().forEach(x -> x.freeRef());
    super._free();
  }
  
  /**
   * Write exploded convolution grid.
   *
   * @param filter the kernel
   * @return the exploded convolution grid
   */
  @Nonnull
  public ExplodedConvolutionGrid write(@Nonnull Tensor filter) {
    if (1 == subLayers.size()) {
      subLayers.get(0).write(filter);
    }
    else {
      for (@Nonnull ExplodedConvolutionLeg leg : subLayers) {
        @Nonnull int[] legDims = {convolutionParams.masterFilterDimensions[0], convolutionParams.masterFilterDimensions[1], leg.getInputBands() * convolutionParams.outputBands};
        @Nonnull Tensor template = new Tensor(legDims);
        @Nullable Tensor tensor = template.mapCoords(c -> {
          int[] coords = c.getCoords();
          return filter.get(coords[0], coords[1], getFilterBand(leg, coords[2]));
        }, false);
        template.freeRef();
        leg.write(tensor);
        tensor.freeRef();
      }
    }
    return this;
  }
  
  /**
   * Read tensor.
   *
   * @param extractor the extractor
   * @return the tensor
   */
  public Tensor read(@Nonnull Function<ExplodedConvolutionLeg, Tensor> extractor) {
    if (1 == subLayers.size()) {
      return extractor.apply(subLayers.get(0));
    }
    else {
      @Nonnull final Tensor filterDelta = new Tensor(convolutionParams.masterFilterDimensions);
      for (@Nonnull ExplodedConvolutionLeg leg : subLayers) {
        Tensor tensor = extractor.apply(leg);
        tensor.forEach((v, c) -> {
          int[] coords = c.getCoords();
          filterDelta.set(coords[0], coords[1], getFilterBand(leg, coords[2]), v);
        }, false);
        tensor.freeRef();
      }
      return filterDelta;
    }
  }
  
  /**
   * Read tensor.
   *
   * @return the tensor
   */
  public Tensor read() {
    return read(l -> l.read());
  }
  
  /**
   * Read tensor.
   *
   * @param deltaSet the delta set
   * @param remove   the remove
   * @return the tensor
   */
  public Tensor read(@Nonnull DeltaSet<Layer> deltaSet, boolean remove) {
    return read(l -> l.read(deltaSet, remove));
  }
  
  private int getFilterBand(@Nonnull ExplodedConvolutionLeg leg, int legFilterBand) {
    int filterBand = legFilterBand;
    filterBand = filterBand + convolutionParams.outputBands * leg.fromBand;
    return filterBand;
  }
  
  /**
   * Gets network.
   *
   * @return the network
   */
  @Nonnull
  public PipelineNetwork getNetwork() {
    assertAlive();
    @Nonnull PipelineNetwork network = new PipelineNetwork(1);
    add(network.getInput(0));
    return network;
  }
  
  /**
   * Add dag node.
   *
   * @param input the input
   * @return the dag node
   */
  public DAGNode add(@Nonnull DAGNode input) {
    assertAlive();
    DAGNetwork network = input.getNetwork();
    int defaultPaddingX = 0;
    int defaultPaddingY = 0;
    boolean customPaddingX = this.convolutionParams.paddingX != null && convolutionParams.paddingX != defaultPaddingX;
    boolean customPaddingY = this.convolutionParams.paddingY != null && convolutionParams.paddingY != defaultPaddingY;
    final DAGNode paddedInput;
    if (customPaddingX || customPaddingY) {
      int x;
      if (this.convolutionParams.paddingX < -defaultPaddingX) {
        x = this.convolutionParams.paddingX + defaultPaddingX;
      }
      else if (this.convolutionParams.paddingX > defaultPaddingX) {
        x = this.convolutionParams.paddingX - defaultPaddingX;
      }
      else {
        x = 0;
      }
      int y;
      if (this.convolutionParams.paddingY < -defaultPaddingY) {
        y = this.convolutionParams.paddingY + defaultPaddingY;
      }
      else if (this.convolutionParams.paddingY > defaultPaddingY) {
        y = this.convolutionParams.paddingY - defaultPaddingY;
      }
      else {
        y = 0;
      }
      if (x != 0 || y != 0) {
        paddedInput = network.wrap(new ImgZeroPaddingLayer(x, y).setPrecision(convolutionParams.precision), input);
      }
      else {
        paddedInput = input;
      }
    }
    else {
      paddedInput = input;
    }
    InnerNode output;
    if (subLayers.size() == 1) {
      output = (InnerNode) subLayers.get(0).add(paddedInput);
    }
    else {
      ImgLinearSubnetLayer linearSubnetLayer = new ImgLinearSubnetLayer();
      subLayers.forEach(leg -> {
        PipelineNetwork subnet = new PipelineNetwork();
        leg.add(subnet.getHead());
        linearSubnetLayer.add(leg.fromBand, leg.toBand, subnet);
      });
      boolean isParallel = CudaSettings.INSTANCE.isConv_para_1();
      linearSubnetLayer.setPrecision(convolutionParams.precision).setParallel(isParallel);
      output = network.wrap(linearSubnetLayer, paddedInput).setParallel(isParallel);
    }
    if (customPaddingX || customPaddingY) {
      int x = !customPaddingX ? 0 : (this.convolutionParams.paddingX - defaultPaddingX);
      int y = !customPaddingY ? 0 : (this.convolutionParams.paddingY - defaultPaddingY);
      if (x > 0) x = 0;
      if (y > 0) y = 0;
      if (x != 0 || y != 0) {
        return network.wrap(new ImgZeroPaddingLayer(x, y).setPrecision(convolutionParams.precision), output);
      }
    }
    return output;
  }
  
  
}
