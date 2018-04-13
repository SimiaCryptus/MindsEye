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
  public static final List<CharSequence> CLASSIC_STYLES = ArtistryUtil.getHadoopFiles("git://github.com/jcjohnson/fast-neural-style.git/master/images/styles/");
  private static final String DEV_BASE = "file:///H:/SimiaCryptus/Artistry/";
  public static final CharSequence threeMusicians = DEV_BASE + "/portraits/picasso/800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
  public static final CharSequence maJolie = DEV_BASE + "/portraits/picasso/Ma_Jolie_Pablo_Picasso.jpg";
  public static final CharSequence monkey = DEV_BASE + "/capuchin-monkey-2759768_960_720.jpg";
  public static final List<CharSequence> picasso = ArtistryUtil.getHadoopFiles(DEV_BASE + "/portraits/picasso/");
  public static final List<CharSequence> vangogh = ArtistryUtil.getHadoopFiles(DEV_BASE + "/portraits/vangogh/");
  public static final List<CharSequence> styles = ArtistryUtil.getHadoopFiles(DEV_BASE + "/styles/");
  public static final List<CharSequence> space = ArtistryUtil.getHadoopFiles(DEV_BASE + "/space/");
  public static final List<CharSequence> michelangelo = ArtistryUtil.getHadoopFiles(DEV_BASE + "/portraits/michelangelo/");
  public static final List<CharSequence> figures = ArtistryUtil.getHadoopFiles(DEV_BASE + "/portraits/figure/");
  public static final List<CharSequence> escher = ArtistryUtil.getHadoopFiles(DEV_BASE + "/portraits/escher/");
  public static final List<CharSequence> waldo = ArtistryUtil.getHadoopFiles(DEV_BASE + "/portraits/waldo/");
  public static final List<CharSequence> owned = ArtistryUtil.getHadoopFiles(DEV_BASE + "/Owned/");
}
