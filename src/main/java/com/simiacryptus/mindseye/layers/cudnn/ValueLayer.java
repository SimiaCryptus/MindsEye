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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * This layer does not require any input, and produces a constant output. This constant can be tuned by optimization
 * processes.
 */
@SuppressWarnings("serial")
public class ValueLayer extends LayerBase {

  private final Precision precision;
  private final CudaTensorList tensorList;

  /**
   * Instantiates a new Const nn layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ValueLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    this.precision = Precision.valueOf(json.get("precision").getAsString());
    Tensor value = Tensor.fromJson(json.get("value"), resources);
    this.tensorList = toDevice(value, precision);
    value.freeRef();
  }

  /**
   * Instantiates a new Const nn layer.
   *
   * @param data the data
   */
  public ValueLayer(final Tensor data) {
    super();
    this.precision = Precision.Float;
    this.tensorList = toDevice(data, precision);
    data.addRef();
    this.frozen = true;
  }

  /**
   * From json const nn layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the const nn layer
   */
  public static ValueLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ValueLayer(json, rs);
  }

  /**
   * To device cuda tensor list.
   *
   * @param data      the data
   * @param precision the precision
   * @return the cuda tensor list
   */
  public CudaTensorList toDevice(final Tensor data, final Precision precision) {
    if (null == data) return null;
    return CudaSystem.run(gpu -> {
      CudaMemory cudaMemory = gpu.allocate(data.length() * precision.size, MemoryType.Managed, true);
      cudaMemory.write(precision, data.getData());
      int[] dimensions = data.getDimensions();
      CudaDevice.CudaTensorDescriptor tensorDescriptor = gpu.newTensorDescriptor(precision, 1, dimensions[2], dimensions[1], dimensions[0]);
      return CudaTensorList.wrap(CudaTensor.wrap(cudaMemory, tensorDescriptor, precision), 1, dimensions, precision);
    });
  }

  @Nonnull
  @Override
  public Result evalAndFree(@Nonnull final Result... array) {
    assert 0 == array.length;
    ValueLayer.this.tensorList.addRef();
    return new Result(tensorList, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
    }) {

      @Override
      protected void _free() {
      }

      @Override
      public boolean isAlive() {
        return false;
      }
    };
  }

  @Override
  protected void _free() {
    tensorList.freeRef();
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    Tensor tensor = tensorList.get(0);
    json.add("value", tensor.toJson(resources, dataSerializer));
    tensor.freeRef();
    json.addProperty("precision", precision.name());
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    Tensor tensor = tensorList.get(0);
    List<double[]> list = Arrays.asList(tensor.getData());
    tensor.freeRef();
    return list;
  }
}
