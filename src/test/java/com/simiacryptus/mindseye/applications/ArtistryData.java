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

package com.simiacryptus.mindseye.applications;

import java.util.Arrays;
import java.util.List;

public class ArtistryData {
  private static final String FAST_NEURAL_STYLE_GIT = "git://github.com/jcjohnson/fast-neural-style.git/master";
  public static final List<CharSequence> CLASSIC_STYLES = ArtistryUtil.getHadoopFiles(FAST_NEURAL_STYLE_GIT + "/images/styles/");
  public static final List<CharSequence> PLANETS = Arrays.asList(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/1024px-The_Earth_seen_from_Apollo_17.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Saturn_during_Equinox.jpg/1920px-Saturn_during_Equinox.jpg"
  );
  public static final List<CharSequence> CLASSIC_CONTENT = ArtistryUtil.getHadoopFiles(FAST_NEURAL_STYLE_GIT + "/images/content/");
}
