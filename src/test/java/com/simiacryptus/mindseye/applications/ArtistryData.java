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

import java.util.List;

public class ArtistryData {
  private static final String FAST_NEURAL_STYLE_GIT = "git://github.com/jcjohnson/fast-neural-style.git/master";
  public static final List<CharSequence> CLASSIC_STYLES = ArtistryUtil.getHadoopFiles(FAST_NEURAL_STYLE_GIT + "/images/styles/");
  public static final List<CharSequence> CLASSIC_CONTENT = ArtistryUtil.getHadoopFiles(FAST_NEURAL_STYLE_GIT + "/images/content/");
  private static final String DEV_BASE = "file:///H:/SimiaCryptus/Artistry/";
  public static final List<CharSequence> picasso = ArtistryUtil.getHadoopFiles(DEV_BASE + "/portraits/picasso/");
  public static final List<CharSequence> vangogh = ArtistryUtil.getHadoopFiles(DEV_BASE + "/portraits/vangogh/");
  public static final List<CharSequence> space = ArtistryUtil.getHadoopFiles(DEV_BASE + "/space/");
  public static final List<CharSequence> owned = ArtistryUtil.getHadoopFiles(DEV_BASE + "/Owned/");
}
