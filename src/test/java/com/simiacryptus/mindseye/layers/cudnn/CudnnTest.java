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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import java.util.ArrayList;

/**
 * The type Cudnn layer run base.
 */
public class CudnnTest extends NotebookReportBase {
  
  /**
   * Instantiates a new Cudnn layer run base.
   */
  public CudnnTest() {
  }
  
  /**
   * Test.
   */
  @Test
  public void test() {
    run(this::test);
  }
  
  private void test(NotebookOutput log) {
    
    log.code(() -> {
      ArrayList<Object> list = new ArrayList<>();
      while (true) {
        list.add(CudaPtr.write(1, Precision.Double, new TensorArray(new Tensor(1024))));
      }
    });
  }
  
  @Override
  public ReportType getReportType() {
    return ReportType.Components;
  }
  
  @Override
  protected Class<?> getTargetClass() {
    return CuDNN.class;
  }
}
