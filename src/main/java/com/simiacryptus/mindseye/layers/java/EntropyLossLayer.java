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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;

/**
 * An entropy-based cost function. The output value is the expected number of nats needed to encode a category chosen
 * using the first input as a distribution, but using the second input distribution for the encoding scheme.
 */
@SuppressWarnings("serial")
public class EntropyLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);

  /**
   * Instantiates a new Entropy loss key.
   */
  public EntropyLossLayer() {
  }

  /**
   * Instantiates a new Entropy loss key.
   *
   * @param id the id
   */
  protected EntropyLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * From json entropy loss key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the entropy loss key
   */
  public static EntropyLossLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new EntropyLossLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    final double zero_tol = 1e-12;
    final Result in0 = inObj[0];
    TensorList indata = in0.getData();
    indata.addRef();
    @Nonnull final Tensor gradient[] = new Tensor[indata.length()];
    final double max_prob = 1.;
    return new Result(TensorArray.wrap(IntStream.range(0, indata.length()).mapToObj(dataIndex -> {
      @Nullable final Tensor l = indata.get(dataIndex);
      @Nullable final Tensor r = inObj[1].getData().get(dataIndex);
      if (l.length() != r.length()) {
        throw new IllegalArgumentException(l.length() + " != " + r.length());
      }
      @Nonnull final Tensor gradientTensor = new Tensor(l.getDimensions());
      @Nullable final double[] gradientData = gradientTensor.getData();
      double total = 0;
      @Nullable final double[] ld = l.getData();
      @Nullable final double[] rd = r.getData();
      for (int i = 0; i < l.length(); i++) {
        final double lv = Math.max(Math.min(ld[i], max_prob), zero_tol);
        final double rv = rd[i];
        if (rv > 0) {
          gradientData[i] = -rv / lv;
          total += -rv * Math.log(lv);
        } else {
          gradientData[i] = 0;
        }
      }
      l.freeRef();
      r.freeRef();
      assert total >= 0;
      gradient[dataIndex] = gradientTensor;
      return new Tensor(new double[]{total}, 1);
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (inObj[1].isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          Tensor deltaTensor = delta.get(dataIndex);
          @Nullable final Tensor inputTensor = indata.get(dataIndex);
          @Nonnull final Tensor passback = new Tensor(gradient[dataIndex].getDimensions());
          for (int i = 0; i < passback.length(); i++) {
            final double lv = Math.max(Math.min(inputTensor.get(i), max_prob), zero_tol);
            passback.set(i, -deltaTensor.get(0) * Math.log(lv));
          }
          inputTensor.freeRef();
          deltaTensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[1].accumulate(buffer, tensorArray);
      }
      if (in0.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          Tensor tensor = delta.get(dataIndex);
          @Nonnull final Tensor passback = new Tensor(gradient[dataIndex].getDimensions());
          for (int i = 0; i < passback.length(); i++) {
            passback.set(i, tensor.get(0) * gradient[dataIndex].get(i));
          }
          tensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        in0.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      protected void _free() {
        indata.freeRef();
        Arrays.stream(gradient).forEach(ReferenceCounting::freeRef);
        Arrays.stream(inObj).forEach(ReferenceCounting::freeRef);
      }

      @Override
      public boolean isAlive() {
        return in0.isAlive() || in0.isAlive();
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
