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

import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.apache.hadoop.yarn.webapp.MimeType;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

/**
 * The type ArtistryAppBase demo.
 */
public abstract class ArtistryAppBase extends NotebookReportBase {
  private static final Logger logger = LoggerFactory.getLogger(ArtistryAppBase.class);
  
  /**
   * The Lake and forest.
   */
  protected final CharSequence lakeAndForest = "H:\\SimiaCryptus\\Artistry\\Owned\\IMG_20170624_153541213-EFFECTS.jpg";
  protected final List<CharSequence> owned = ArtistryUtil.getFiles("H:\\SimiaCryptus\\Artistry\\Owned\\");
  /**
   * The Monkey.
   */
  protected final String monkey = "H:\\SimiaCryptus\\Artistry\\capuchin-monkey-2759768_960_720.jpg";
  /**
   * The Van gogh 1.
   */
  protected final CharSequence vanGogh1 = "H:\\SimiaCryptus\\Artistry\\portraits\\vangogh\\Van_Gogh_-_Portrait_of_Pere_Tanguy_1887-8.jpg";
  /**
   * The Van gogh 2.
   */
  protected final CharSequence vanGogh2 = "H:\\SimiaCryptus\\Artistry\\portraits\\vangogh\\800px-Vincent_van_Gogh_-_Dr_Paul_Gachet_-_Google_Art_Project.jpg";
  /**
   * The Three musicians.
   */
  protected final CharSequence threeMusicians = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
  /**
   * The Ma jolie.
   */
  protected final CharSequence maJolie = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\Ma_Jolie_Pablo_Picasso.jpg";
  /**
   * The Picasso.
   */
  protected final List<CharSequence> picasso = ArtistryUtil.getFiles("H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\");
  /**
   * The Vangogh.
   */
  protected final List<CharSequence> vangogh = ArtistryUtil.getFiles("H:\\SimiaCryptus\\Artistry\\portraits\\vangogh\\");
  /**
   * The Michelangelo.
   */
  protected final List<CharSequence> michelangelo = ArtistryUtil.getFiles("H:\\SimiaCryptus\\Artistry\\portraits\\michelangelo\\");
  /**
   * The Figures.
   */
  protected final List<CharSequence> figures = ArtistryUtil.getFiles("H:\\SimiaCryptus\\Artistry\\portraits\\figure\\");
  /**
   * The Escher.
   */
  protected final List<CharSequence> escher = ArtistryUtil.getFiles("H:\\SimiaCryptus\\Artistry\\portraits\\escher\\");
  /**
   * The Waldo.
   */
  protected final List<CharSequence> waldo = ArtistryUtil.getFiles("H:\\SimiaCryptus\\Artistry\\portraits\\waldo\\");
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public final void run() {
    run(notebookOutput -> run(notebookOutput), getClass().getSimpleName() + "_" + new SimpleDateFormat("yyyyMMddHHmm").format(new Date()));
  }
  
  /**
   * Run.
   *
   * @param notebookOutput the notebook output
   */
  protected abstract void run(final NotebookOutput notebookOutput);
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
  }
  
  /**
   * Init.
   *
   * @param log the log
   */
  public void init(final NotebookOutput log) {
    log.getHttpd().addHandler("gpu.json", MimeType.JSON, out -> {
      try {
        JsonUtil.MAPPER.writer().writeValue(out, CudaSystem.getExecutionStatistics());
        //JsonUtil.MAPPER.writer().writeValue(out, new HashMap<>());
        out.close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
    //server.dataReciever
    //server.init();
    //server.start();
    @Nonnull String logName = "cuda_" + log.getName() + ".log";
    log.p(log.file((String) null, logName, "GPU Log"));
    CudaSystem.addLog(new PrintStream(log.file(logName)));
  }
}
