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
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

/**
 * Rectified Linear Unit. y=(x&lt;0)?0:x
 */
@SuppressWarnings("serial")
public class ReLuActivationLayer extends LayerBase {


  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ReLuActivationLayer.class);
  @Nullable
  private final Tensor weights;

  /**
   * Instantiates a new Re lu activation key.
   */
  public ReLuActivationLayer() {
    super();
    weights = new Tensor(1);
    weights.set(0, 1.);
    this.frozen = true;
  }

  /**
   * Instantiates a new Re lu activation key.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ReLuActivationLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
  }

  /**
   * From json re lu activation key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the re lu activation key
   */
  public static ReLuActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ReLuActivationLayer(json, rs);
  }

  @Override
  protected void _free() {
    weights.freeRef();
    super._free();
  }

  /**
   * Add weights re lu activation key.
   *
   * @param f the f
   * @return the re lu activation key
   */
  @Nonnull
  public ReLuActivationLayer addWeights(@Nonnull final DoubleSupplier f) {
    Util.add(f, weights.getData());
    return this;
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result input = inObj[0];
    final TensorList indata = input.getData();
    input.addRef();
    indata.addRef();
    weights.addRef();
    final int itemCnt = indata.length();
    return new Result(TensorArray.wrap(IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
      @Nullable Tensor tensorElement = indata.get(dataIndex);
      @Nonnull final Tensor tensor = tensorElement.multiply(weights.get(0));
      tensorElement.freeRef();
      @Nullable final double[] outputData = tensor.getData();
      for (int i = 0; i < outputData.length; i++) {
        if (outputData[i] < 0) {
          outputData[i] = 0;
        }
      }
      return tensor;
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (!isFrozen()) {
        IntStream.range(0, delta.length()).parallel().forEach(dataIndex -> {
          @Nullable Tensor deltaTensor = delta.get(dataIndex);
          @Nullable final double[] deltaData = deltaTensor.getData();
          @Nullable Tensor inputTensor = indata.get(dataIndex);
          @Nullable final double[] inputData = inputTensor.getData();
          @Nonnull final Tensor weightDelta = new Tensor(weights.getDimensions());
          @Nullable final double[] weightDeltaData = weightDelta.getData();
          for (int i = 0; i < deltaData.length; i++) {
            weightDeltaData[0] += inputData[i] < 0 ? 0 : deltaData[i] * inputData[i];
          }
          buffer.get(ReLuActivationLayer.this.getId(), weights.getData()).addInPlace(weightDeltaData).freeRef();
          deltaTensor.freeRef();
          inputTensor.freeRef();
          weightDelta.freeRef();
        });
      }
      if (input.isAlive()) {
        final double weight = weights.getData()[0];
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).parallel().mapToObj(dataIndex -> {
          @Nullable Tensor deltaTensor = delta.get(dataIndex);
          @Nullable final double[] deltaData = deltaTensor.getData();
          @Nullable Tensor inTensor = indata.get(dataIndex);
          @Nullable final double[] inputData = inTensor.getData();
          @Nonnull final int[] dims = inTensor.getDimensions();
          @Nonnull final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.length(); i++) {
            passback.set(i, inputData[i] < 0 ? 0 : deltaData[i] * weight);
          }
          inTensor.freeRef();
          deltaTensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      protected void _free() {
        input.freeRef();
        indata.freeRef();
        weights.freeRef();
      }

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("weights", weights.toJson(resources, dataSerializer));
    return json;
  }

  /**
   * Gets mobility.
   *
   * @return the mobility
   */
  protected double getMobility() {
    return 1;
  }

  /**
   * Sets weight.
   *
   * @param data the data
   * @return the weight
   */
  @Nonnull
  public ReLuActivationLayer setWeight(final double data) {
    weights.set(0, data);
    return this;
  }

  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @Nonnull
  public ReLuActivationLayer setWeights(@Nonnull final DoubleSupplier f) {
    Arrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
    return this;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(weights.getData());
  }

}
