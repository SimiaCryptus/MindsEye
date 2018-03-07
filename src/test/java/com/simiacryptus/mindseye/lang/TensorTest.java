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

package com.simiacryptus.mindseye.lang;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Assert;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Tensor eval.
 */
public class TensorTest {
  private static final Logger log = LoggerFactory.getLogger(TensorTest.class);
  
  /**
   * Parse tensor.
   *
   * @param str the str
   * @return the tensor
   */
  @Nullable
  public Tensor parse(final String str) {
    final JsonElement json = new GsonBuilder().create().fromJson(str, JsonElement.class);
    @Nullable final Tensor tensor = Tensor.fromJson(json, null);
    Assert.assertEquals(json, tensor.toJson(null, Tensor.json_precision));
    return tensor;
  }
  
  /**
   * Test.
   *
   * @param t the t
   */
  public void test(@Nonnull final Tensor t) {
    @Nonnull final JsonElement json = t.toJson(null, Tensor.json_precision);
    Assert.assertEquals(Tensor.fromJson(json, null), t);
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
    final List<String> coordinates = new Tensor(2, 2, 2).coordStream(true)
      .map(c -> String.format("%s - %s", c.getIndex(), Arrays.toString(c.getCoords()))).collect(Collectors.toList());
    for (final String c : coordinates) {
      log.info(c);
    }
  }
  
  /**
   * Test shuffle stream.
   *
   * @throws Exception the exception
   */
  @Test
  @Category(TestCategories.UnitTest.class)
  public void testShuffleStream() throws Exception {
    @Nonnull HashSet<Object> ids = new HashSet<>();
    int max = 10000;
    TestUtil.shuffle(IntStream.range(0, max)).forEach((int i) -> {
      if (i >= 0 && i >= max) throw new AssertionError(i);
      if (!ids.add(i)) throw new AssertionError(i);
    });
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
