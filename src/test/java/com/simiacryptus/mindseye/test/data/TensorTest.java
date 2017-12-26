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

package com.simiacryptus.mindseye.test.data;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Assert;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The type Tensor run.
 */
public class TensorTest {
  private static final Logger logger = LoggerFactory.getLogger(TensorTest.class);
  
  /**
   * Parse tensor.
   *
   * @param str the str
   * @return the tensor
   */
  public Tensor parse(final String str) {
    final JsonElement json = new GsonBuilder().create().fromJson(str, JsonElement.class);
    final Tensor tensor = Tensor.fromJson(json);
    Assert.assertEquals(json, tensor.toJson());
    return tensor;
  }
  
  /**
   * Test.
   *
   * @param t the t
   */
  public void test(final Tensor t) {
    final JsonElement json = t.toJson();
    Assert.assertEquals(Tensor.fromJson(json), t);
    parse(json.toString());
  }
  
  /**
   * Test coord stream.
   *
   * @throws Exception the exception
   */
  @Test
  @Category(TestCategories.UnitTest.class)
  public void testCoordStream() throws Exception {
    final List<String> coordinates = new Tensor(2, 2, 2).coordStream()
      .map(c -> String.format("%s - %s", c.getIndex(), Arrays.toString(c.getCoords()))).collect(Collectors.toList());
    for (final String c : coordinates) {
      logger.info(c);
    }
  }
  
  /**
   * Test to json.
   *
   * @throws Exception the exception
   */
  @Test
  @Category(TestCategories.UnitTest.class)
  public void testToJson() throws Exception {
    test(new Tensor(3, 3, 1).map(v -> Math.random()));
    test(new Tensor(1, 3, 3).map(v -> Math.random()));
  }
  
}
