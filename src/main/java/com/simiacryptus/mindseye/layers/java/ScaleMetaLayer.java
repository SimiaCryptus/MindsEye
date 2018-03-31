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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The type Scale meta layer.
 */
@SuppressWarnings("serial")
public class ScaleMetaLayer extends LayerBase {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ScaleMetaLayer.class);
  
  /**
   * Instantiates a new Scale meta layer.
   */
  public ScaleMetaLayer() {
  }
  
  /**
   * Instantiates a new Scale meta layer.
   *
   * @param id the id
   */
  protected ScaleMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json scale meta layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the scale meta layer
   */
  public static ScaleMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ScaleMetaLayer(json);
  }
  
  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final int itemCnt = inObj[0].getData().length();
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    final Tensor[] tensors = IntStream.range(0, itemCnt).mapToObj(dataIndex -> inObj[0].getData().get(dataIndex).mapIndex((v, c) -> v * inObj[1].getData().get(0).get(c))).toArray(i -> new Tensor[i]);
    Tensor tensor0 = tensors[0];
    tensor0.addRef();
    return new Result(TensorArray.wrap(tensors), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
      if (inObj[0].isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(data.stream().map(t -> {
          @Nullable Tensor t1 = inObj[1].getData().get(0);
          @Nullable Tensor tensor = t.mapIndex((v, c) -> {
            return v * t1.get(c);
          });
          t.freeRef();
          t1.freeRef();
          return tensor;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorArray);
      }
      if (inObj[1].isAlive()) {
        @Nullable final Tensor passback = tensor0.mapIndex((v, c) -> {
          return IntStream.range(0, itemCnt).mapToDouble(i -> data.get(i).get(c) * inObj[0].getData().get(i).get(c)).sum();
        });
        tensor0.freeRef();
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, inObj[1].getData().length())
          .mapToObj(i -> i == 0 ? passback : passback.map(v -> 0)).toArray(i -> new Tensor[i]));
        inObj[1].accumulate(buffer, tensorArray);
      }
      Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || inObj[1].isAlive();
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
