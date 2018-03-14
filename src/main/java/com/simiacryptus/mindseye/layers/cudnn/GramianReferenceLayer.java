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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * A dense matrix operator using vector-matrix multiplication. Represents a fully connected layer of synapses, where all
 * inputs are connected to all outputs via seperate coefficients.
 */
@SuppressWarnings("serial")
public class GramianReferenceLayer extends LayerBase implements MultiPrecision<GramianReferenceLayer> {
  private static final Logger log = LoggerFactory.getLogger(GramianReferenceLayer.class);
  
  
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img concat layer.
   */
  public GramianReferenceLayer() {
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected GramianReferenceLayer(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static GramianReferenceLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new GramianReferenceLayer(json, rs);
  }
  
  @Nullable
  @Override
  public Result evalAndFree(final Result... inObj) {
    assert 1 == inObj.length;
    TensorList inputData = inObj[0].getData();
    int[] dimensions = inputData.getDimensions();
    assert 3 == dimensions.length;
    
    PipelineNetwork network = new PipelineNetwork();
    DAGNode input = network.getInput(0);
    network.wrap(new ImgConcatLayer().setParallel(false), IntStream.range(0, dimensions[2]).mapToObj(band -> {
      return network.wrap(new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg),
        network.wrap(new GateProductLayer(), input,
          network.wrap(new ImgBandSelectLayer(band, band + 1), input)
        ));
    }).toArray(i -> new DAGNode[i]));
    Result result = network.evalAndFree(inObj);
    network.freeRef();
    
    
    return result;
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public GramianReferenceLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
}
