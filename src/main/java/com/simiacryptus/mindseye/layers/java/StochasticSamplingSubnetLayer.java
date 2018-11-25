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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.StochasticComponent;
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;
import com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer;
import com.simiacryptus.mindseye.network.CountingResult;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.IntStream;

/**
 * This key works as a scaling function, similar to a father wavelet. Allows convolutional and pooling layers to work
 * across larger png regions.
 */
@SuppressWarnings("serial")
public class StochasticSamplingSubnetLayer extends LayerBase implements StochasticComponent {

  private final int samples;
  @Nullable
  private final Layer subnetwork;
  private long seed = System.nanoTime();
  private long layerSeed = System.nanoTime();

  /**
   * Instantiates a new Rescaled subnet key.
   *
   * @param subnetwork the subnetwork
   * @param samples    the samples
   */
  public StochasticSamplingSubnetLayer(final Layer subnetwork, final int samples) {
    super();
    this.samples = samples;
    this.subnetwork = subnetwork;
    this.subnetwork.addRef();
  }

  /**
   * Instantiates a new Rescaled subnet key.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected StochasticSamplingSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    samples = json.getAsJsonPrimitive("samples").getAsInt();
    seed = json.getAsJsonPrimitive("seed").getAsInt();
    layerSeed = json.getAsJsonPrimitive("layerSeed").getAsInt();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
    this.subnetwork = subnetwork == null ? null : Layer.fromJson(subnetwork, rs);
  }

  /**
   * From json rescaled subnet key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the rescaled subnet key
   */
  public static StochasticSamplingSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new StochasticSamplingSubnetLayer(json, rs);
  }

  /**
   * Average result.
   *
   * @param samples the samples
   * @return the result
   */
  public static Result average(final Result[] samples) {
    PipelineNetwork gateNetwork = new PipelineNetwork(1);
    gateNetwork.wrap(new ProductLayer(),
        gateNetwork.getInput(0),
        gateNetwork.wrap(new ValueLayer(new Tensor(1, 1, 1).mapAndFree(v -> 1.0 / samples.length)), new DAGNode[]{})).freeRef();
    SumInputsLayer sumInputsLayer = new SumInputsLayer();
    try {
      return gateNetwork.evalAndFree(sumInputsLayer.evalAndFree(samples));
    } finally {
      sumInputsLayer.freeRef();
      gateNetwork.freeRef();
    }
  }

  @Override
  protected void _free() {
    this.subnetwork.freeRef();
    super._free();
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    Result[] counting = Arrays.stream(inObj).map(r -> {
      return new CountingResult(r, samples);
    }).toArray(i -> new Result[i]);
    return average(Arrays.stream(getSeeds()).mapToObj(seed -> {
      if (subnetwork instanceof DAGNetwork) {
        ((DAGNetwork) subnetwork).visitNodes(node -> {
          Layer layer = node.getLayer();
          if (layer instanceof StochasticComponent) {
            ((StochasticComponent) layer).shuffle(seed);
          }
        });
      }
      if (subnetwork instanceof StochasticComponent) {
        ((StochasticComponent) subnetwork).shuffle(seed);
      }
      return subnetwork.eval(counting);
    }).toArray(i -> new Result[i]));
  }

  /**
   * Get seeds long [ ].
   *
   * @return the long [ ]
   */
  public long[] getSeeds() {
    Random random = new Random(seed + layerSeed);
    return IntStream.range(0, this.samples).mapToLong(i -> random.nextLong()).toArray();
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("samples", samples);
    json.addProperty("seed", seed);
    json.addProperty("layerSeed", layerSeed);
    json.add("subnetwork", subnetwork.getJson(resources, dataSerializer));
    return json;
  }


  @Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }

  @Nonnull
  @Override
  public Layer setFrozen(final boolean frozen) {
    subnetwork.setFrozen(frozen);
    return super.setFrozen(frozen);
  }

  @Override
  public void shuffle(final long seed) {
    this.seed = seed;
  }

  @Override
  public void clearNoise() {
    seed = 0;
  }

}
