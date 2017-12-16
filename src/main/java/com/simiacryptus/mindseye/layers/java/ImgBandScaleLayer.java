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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;
import java.util.stream.IntStream;

/**
 * The type Img band scale layer.
 */
public class ImgBandScaleLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgBandScaleLayer.class);
  private final double[] wieghts;
  
  /**
   * Instantiates a new Img band scale layer.
   *
   * @param json the json
   */
  protected ImgBandScaleLayer(JsonObject json) {
    super(json);
    this.wieghts = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }
  
  /**
   * Instantiates a new Img band scale layer.
   */
  protected ImgBandScaleLayer() {
    super();
    this.wieghts = null;
  }
  
  
  /**
   * Instantiates a new Img band scale layer.
   *
   * @param bands the bands
   */
  public ImgBandScaleLayer(final double... bands) {
    super();
    this.wieghts = bands;
  }
  
  /**
   * From json img band scale layer.
   *
   * @param json the json
   * @return the img band scale layer
   */
  public static ImgBandScaleLayer fromJson(JsonObject json) {
    return new ImgBandScaleLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getWieghts()));
    return json;
  }
  
  private double[] fn(final double[] input) {
    assert Arrays.stream(input).allMatch(v -> Double.isFinite(v));
    assert (null != input);
    double[] wieghts = this.getWieghts();
    assert (null != wieghts);
    if (input.length % wieghts.length != 0) throw new IllegalArgumentException();
    final double[] array = new double[input.length];
    int size = input.length / wieghts.length;
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] * wieghts[i / size];
    }
    assert Arrays.stream(array).allMatch(v -> Double.isFinite(v));
    return array;
  }
  
  /**
   * Add weights img band scale layer.
   *
   * @param f the f
   * @return the img band scale layer
   */
  public ImgBandScaleLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.getWieghts());
    return this;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    return eval(inObj[0]);
  }
  
  /**
   * Eval nn result.
   *
   * @param input the input
   * @return the nn result
   */
  public NNResult eval(NNResult input) {
    final double[] bias = getWieghts();
    assert input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
    Tensor[] outputA = input.getData().stream().parallel()
      .map(r -> {
        if (r.getDimensions().length != 3) {
          throw new IllegalArgumentException(Arrays.toString(r.getDimensions()));
        }
        if (r.getDimensions()[2] != bias.length) {
          throw new IllegalArgumentException(String.format("%s: %s does not have %s bands",
            getName(), Arrays.toString(r.getDimensions()), bias.length));
        }
        return new Tensor(fn(r.getData()), r.getDimensions());
      })
      .toArray(i -> new Tensor[i]);
    assert Arrays.stream(outputA).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        assert data.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        if (!isFrozen()) {
          Delta<NNLayer> deltaBuffer = buffer.get(ImgBandScaleLayer.this, bias);
          IntStream.range(0, data.length()).forEach(index -> {
            final double[] array = RecycleBin.DOUBLES.obtain(bias.length);
            double[] signal = data.get(index).getData();
            double[] in = input.getData().get(index).getData();
            int size = signal.length / bias.length;
            for (int i = 0; i < signal.length; i++) {
              array[i / size] += signal[i] * in[i];
              if (!Double.isFinite(array[i / size])) array[i / size] = 0.0;
            }
            assert Arrays.stream(array).allMatch(v -> Double.isFinite(v));
            deltaBuffer.addInPlace(array);
            RecycleBin.DOUBLES.recycle(array);
          });
        }
        if (input.isAlive()) {
          input.accumulate(buffer, new TensorArray(data.stream().map(
            t -> t.mapCoords((c) -> t.get(c) * bias[c.getCoords()[2]])
          ).toArray(i -> new Tensor[i])));
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  /**
   * Set nn layer.
   *
   * @param ds the ds
   * @return the nn layer
   */
  public NNLayer set(final double[] ds) {
    double[] bias = this.getWieghts();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    assert Arrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public ImgBandScaleLayer setWeights(final IntToDoubleFunction f) {
    double[] bias = this.getWieghts();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    assert Arrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.getWieghts());
  }
  
  /**
   * Get wieghts double [ ].
   *
   * @return the double [ ]
   */
  public double[] getWieghts() {
    if (!Arrays.stream(wieghts).allMatch(v -> Double.isFinite(v))) {
      throw new IllegalStateException(Arrays.toString(wieghts));
    }
    return wieghts;
  }
}
