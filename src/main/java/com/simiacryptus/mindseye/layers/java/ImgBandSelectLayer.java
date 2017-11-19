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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class ImgBandSelectLayer extends NNLayer {
  
  
  private final int[] bands;
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    JsonArray array = new JsonArray();
    for(int b : bands) array.add(new JsonPrimitive(b));
    json.add("bands", array);
    return json;
  }
  
  /**
   * From json img reshape layer.
   *
   * @param json the json
   * @return the img reshape layer
   */
  public static ImgBandSelectLayer fromJson(JsonObject json) {
    return new ImgBandSelectLayer(json);
  }
  
  /**
   * Instantiates a new Img reshape layer.
   *
   * @param json the json
   */
  protected ImgBandSelectLayer(JsonObject json) {
    super(json);
    JsonArray jsonArray = json.getAsJsonArray("bands");
    this.bands = new int[jsonArray.size()];
    for(int i=0;i<bands.length;i++) bands[i] = jsonArray.get(i).getAsInt();
  }
  
  /**
   * Instantiates a new Img reshape layer.
   *
   */
  public ImgBandSelectLayer(int... bands) {
    super();
    this.bands = bands;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputDims = batch.get(0).getDimensions();
    assert (3 == inputDims.length);
    Tensor outputDims = new Tensor(inputDims[0], inputDims[1], bands.length);
    return new NNResult(IntStream.range(0, batch.length()).parallel()
                          .mapToObj(dataIndex -> outputDims.mapCoords((v,c)-> batch.get(dataIndex).get(c.coords[0],c.coords[1],bands[c.coords[2]])))
                          .toArray(i -> new Tensor[i])) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList error) {
        if (input.isAlive()) {
          final Tensor[] data1 = IntStream.range(0, error.length()).parallel()
                                   .mapToObj(dataIndex -> {
                                     Tensor passback = new Tensor(inputDims);
                                     Tensor err = error.get(dataIndex);
                                     err.coordStream().forEach(c->{
                                       passback.set(c.coords[0],c.coords[1],bands[c.coords[2]], err.get(c));
                                     });
                                     return passback;
                                   }).toArray(i -> new Tensor[i]);
          input.accumulate(buffer, new TensorArray(data1));
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  
}
