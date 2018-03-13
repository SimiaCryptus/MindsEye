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

import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

/**
 * The type Tensor apply.
 */
public class DataSerializerTest {
  private static final Logger log = LoggerFactory.getLogger(DataSerializerTest.class);
  
  /**
   * Test coord stream.
   *
   * @throws Exception the exception
   */
  @Test
  @Category(TestCategories.UnitTest.class)
  public void testDouble() throws Exception {
    test(SerialPrecision.Double);
  }
  
  /**
   * Test float.
   *
   * @throws Exception the exception
   */
  @Test
  @Category(TestCategories.UnitTest.class)
  public void testFloat() throws Exception {
    test(SerialPrecision.Float);
  }
  
  /**
   * Test uniform 32.
   *
   * @throws Exception the exception
   */
  @Test
  @Category(TestCategories.UnitTest.class)
  public void testUniform32() throws Exception {
    test(SerialPrecision.Uniform32);
  }
  
  /**
   * Test uniform 16.
   *
   * @throws Exception the exception
   */
  @Test
  @Category(TestCategories.UnitTest.class)
  public void testUniform16() throws Exception {
    test(SerialPrecision.Uniform16);
  }
  
  /**
   * Test uniform 8.
   *
   * @throws Exception the exception
   */
  @Test
  @Category(TestCategories.UnitTest.class)
  public void testUniform8() throws Exception {
    test(SerialPrecision.Uniform8);
  }
  
  /**
   * Test.
   *
   * @param target the target
   */
  public void test(@Nonnull DataSerializer target) {
    test(target, this::random1, "Uniform");
    test(target, this::random2, "Exponential");
  }
  
  /**
   * Test.
   *
   * @param target the target
   * @param f      the f
   * @param name   the name
   */
  public void test(@Nonnull DataSerializer target, @Nonnull DoubleSupplier f, String name) {
    @Nonnull double[] source = random(1024, f);
    @Nonnull double[] result = target.fromBytes(target.toBytes(source));
    double rms = IntStream.range(0, source.length).mapToDouble(i -> (source[i] - result[i]) / (source[i] + result[i])).map(x -> x * x).average().getAsDouble();
    log.info(String.format("%s RMS: %s", name, rms));
    //assert rms < 1e-4;
  }
  
  @Nonnull
  private double[] random(int i, @Nonnull DoubleSupplier f) {
    @Nonnull double[] doubles = new double[i];
    Arrays.parallelSetAll(doubles, j -> f.getAsDouble());
    return doubles;
  }
  
  private double random1() {
    return Math.random();
  }
  
  private double random2() {
    return Math.exp(Math.random());
  }
  
}
