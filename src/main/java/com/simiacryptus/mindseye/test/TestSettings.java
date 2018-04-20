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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.Settings;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;

/**
 * The type Cuda settings.
 */
public class TestSettings implements Settings {
  /**
   * The constant INSTANCE.
   */
  public static final TestSettings INSTANCE = new TestSettings();
  /**
   * The Tag.
   */
  public final String tag;
  
  public boolean autobrowse;
  
  private TestSettings() {
    if (CudaSettings.INSTANCE == null) throw new RuntimeException();
    tag = Settings.get("GIT_TAG", "master");
    autobrowse = false;
  }
  
}
