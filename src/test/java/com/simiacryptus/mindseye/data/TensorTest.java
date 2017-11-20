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

package com.simiacryptus.mindseye.data;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The type Tensor run.
 */
public class TensorTest {
  /**
   * Test coord stream.
   *
   * @throws Exception the exception
   */
  @Test
  @Category(TestCategories.UnitTest.class)
  public void testCoordStream() throws Exception {
    List<String> coordinates = new Tensor(2, 2, 2).coordStream()
      .map(c -> String.format("%s - %s", c.index, Arrays.toString(c.coords))).collect(Collectors.toList());
    for (String c : coordinates) {
      System.out.println(c);
    }
  }
  
}
