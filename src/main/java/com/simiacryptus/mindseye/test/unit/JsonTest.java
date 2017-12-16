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

package com.simiacryptus.mindseye.test.unit;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.io.NotebookOutput;

/**
 * The type Json test.
 */
public class JsonTest implements ComponentTest {
  @Override
  public ToleranceStatistics test(NotebookOutput log, NNLayer layer, Tensor... inputPrototype) {
    log.h3("Json Serialization");
    log.code(() -> {
      JsonObject json = layer.getJson();
      NNLayer echo = NNLayer.fromJson(json);
      if ((echo == null)) throw new AssertionError("Failed to deserialize");
      if ((layer == echo)) throw new AssertionError("Serialization did not copy");
      if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
      return new GsonBuilder().setPrettyPrinting().create().toJson(json);
    });
    return null;
  }
}
