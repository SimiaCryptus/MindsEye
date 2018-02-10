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

package com.simiacryptus.mindseye.test.unit;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;
import org.apache.commons.io.IOUtils;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.HashMap;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

/**
 * The type Json run.
 */
public class SerializationTest extends ComponentTestBase<ToleranceStatistics> {
  @Nonnull
  private final HashMap<SerialPrecision, NNLayer> models = new HashMap<>();
  private boolean persist = false;
  
  /**
   * Compress gz byte [ ].
   *
   * @param prettyPrint the pretty print
   * @return the byte [ ]
   */
  public static byte[] compressGZ(@javax.annotation.Nonnull String prettyPrint) {
    return compressGZ(prettyPrint.getBytes(Charset.forName("UTF-8")));
  }
  
  /**
   * Compress gz byte [ ].
   *
   * @param bytes the bytes
   * @return the byte [ ]
   */
  public static byte[] compressGZ(byte[] bytes) {
    @javax.annotation.Nonnull ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    try {
      try (@javax.annotation.Nonnull GZIPOutputStream out = new GZIPOutputStream(byteArrayOutputStream)) {
        IOUtils.write(bytes, out);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return byteArrayOutputStream.toByteArray();
  }
  
  @Nullable
  @Override
  public ToleranceStatistics test(@javax.annotation.Nonnull final NotebookOutput log, @javax.annotation.Nonnull final NNLayer layer, final Tensor... inputPrototype) {
    log.h1("Serialization");
    log.p("This run will demonstrate the layer's JSON serialization, and verify deserialization integrity.");
    
    String prettyPrint = "";
    log.h2("Raw Json");
    try {
      prettyPrint = log.code(() -> {
        final JsonObject json = layer.getJson();
        @javax.annotation.Nonnull final NNLayer echo = NNLayer.fromJson(json);
        if (echo == null) throw new AssertionError("Failed to deserialize");
        if (layer == echo) throw new AssertionError("Serialization did not copy");
        if (!layer.equals(echo)) throw new AssertionError("Serialization not equal");
        echo.freeRef();
        return new GsonBuilder().setPrettyPrinting().create().toJson(json);
      });
      @javax.annotation.Nonnull String filename = layer.getClass().getSimpleName() + "_" + log.getName() + ".json";
      log.p(log.file(prettyPrint, filename, String.format("Wrote Model to %s; %s characters", filename, prettyPrint.length())));
    } catch (RuntimeException e) {
      e.printStackTrace();
      Util.sleep(1000);
    } catch (OutOfMemoryError e) {
      e.printStackTrace();
      Util.sleep(1000);
    }
    log.p("");
    @javax.annotation.Nonnull Object outSync = new Object();
    if (prettyPrint.isEmpty() || prettyPrint.length() > 1024 * 64)
      Arrays.stream(SerialPrecision.values()).parallel().forEach(precision -> {
        try {
          @javax.annotation.Nonnull File file = new File(log.getResourceDir(), log.getName() + "_" + precision.name() + ".zip");
          layer.writeZip(file, precision);
          @javax.annotation.Nonnull final NNLayer echo = NNLayer.fromZip(new ZipFile(file));
          getModels().put(precision, echo);
          synchronized (outSync) {
            log.h2(String.format("Zipfile %s", precision.name()));
            log.p(log.link(file, String.format("Wrote Model with %s precision to %s; %.3fMiB bytes", precision, file.getName(), file.length() * 1.0 / (0x100000))));
          }
          if (!isPersist()) file.delete();
          if (echo == null) throw new AssertionError("Failed to deserialize");
          if (layer == echo) throw new AssertionError("Serialization did not copy");
          if (!layer.equals(echo)) throw new AssertionError("Serialization not equal");
          echo.freeRef();
        } catch (RuntimeException e) {
          e.printStackTrace();
        } catch (OutOfMemoryError e) {
          e.printStackTrace();
        } catch (ZipException e) {
          e.printStackTrace();
        } catch (IOException e) {
          e.printStackTrace();
        }
      });
    
    return null;
  }
  
  /**
   * Gets models.
   *
   * @return the models
   */
  @javax.annotation.Nonnull
  public HashMap<SerialPrecision, NNLayer> getModels() {
    return models;
  }
  
  /**
   * Is persist boolean.
   *
   * @return the boolean
   */
  public boolean isPersist() {
    return persist;
  }
  
  /**
   * Sets persist.
   *
   * @param persist the persist
   * @return the persist
   */
  @javax.annotation.Nonnull
  public SerializationTest setPersist(boolean persist) {
    this.persist = persist;
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public String toString() {
    return "SerializationTest{" +
      "models=" + models +
      ", persist=" + persist +
      '}';
  }
}
