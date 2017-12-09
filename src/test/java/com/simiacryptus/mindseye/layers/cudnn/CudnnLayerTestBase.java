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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.IOException;
import java.io.PrintStream;
import java.util.zip.GZIPOutputStream;

public abstract class CudnnLayerTestBase extends LayerTestBase {
  
  @Override
  public void test(NotebookOutput log) {
    try {
      CuDNN.apiLog = new PrintStream(log.file("cuda.log"));
      super.test(log);
    } finally {
      log.p(log.file(null,"cuda.log","GPU Log"));
      CuDNN.apiLog.close();
      CuDNN.apiLog = null;
    }
  }
  
}
