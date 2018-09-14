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
import com.simiacryptus.mindseye.lang.Layer;
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
 * The type Json apply.
 */
public class SerializationTest extends ComponentTestBase<ToleranceStatistics> {
  @Nonnull
  private final HashMap<SerialPrecision, Layer> models = new HashMap<>();
  private boolean persist = false;
  
  /**
   * Compress gz byte [ ].
   *
   * @param prettyPrint the pretty print
   * @return the byte [ ]
   */
  public static byte[] compressGZ(@Nonnull String prettyPrint) {
    return compressGZ(prettyPrint.getBytes(Charset.forName("UTF-8")));
  }
  
  /**
   * Compress gz byte [ ].
   *
   * @param bytes the bytes
   * @return the byte [ ]
   */
  public static byte[] compressGZ(byte[] bytes) {
    @Nonnull ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    try {
      try (@Nonnull GZIPOutputStream out = new GZIPOutputStream(byteArrayOutputStream)) {
        IOUtils.write(bytes, out);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return byteArrayOutputStream.toByteArray();
  }
  
  @Nullable
  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, @Nonnull final Layer layer, final Tensor... inputPrototype) {
    log.h1("Serialization");
    log.p("This apply will demonstrate the layer's JSON serialization, and verify deserialization integrity.");
    
    String prettyPrint = "";
    log.h2("Raw Json");
    try {
      prettyPrint = log.eval(() -> {
        final JsonObject json = layer.getJson();
        @Nonnull final Layer echo = Layer.fromJson(json);
        if (echo == null) throw new AssertionError("Failed to deserialize");
        if (layer == echo) throw new AssertionError("Serialization did not copy");
        if (!layer.equals(echo)) throw new AssertionError("Serialization not equal");
        echo.freeRef();
        return new GsonBuilder().setPrettyPrinting().create().toJson(json);
      });
      @Nonnull String filename = layer.getClass().getSimpleName() + "_" + log.getName() + ".json";
      log.p(log.file(prettyPrint, filename, String.format("Wrote Model to %s; %s characters", filename, prettyPrint.length())));
    } catch (RuntimeException e) {
      e.printStackTrace();
      Util.sleep(1000);
    } catch (OutOfMemoryError e) {
      e.printStackTrace();
      Util.sleep(1000);
    }
    log.p("");
    @Nonnull Object outSync = new Object();
    if (prettyPrint.isEmpty() || prettyPrint.length() > 1024 * 64)
      Arrays.stream(SerialPrecision.values()).parallel().forEach(precision -> {
        try {
          @Nonnull File file = new File(log.getResourceDir(), log.getName() + "_" + precision.name() + ".zip");
          layer.writeZip(file, precision);
          @Nonnull final Layer echo = Layer.fromZip(new ZipFile(file));
          getModels().put(precision, echo);
          synchronized (outSync) {
            log.h2(String.format("Zipfile %s", precision.name()));
            log.p(log.link(file, String.format("Wrote Model apply %s precision to %s; %.3fMiB bytes", precision, file.getName(), file.length() * 1.0 / (0x100000))));
          }
          if (!isPersist()) file.delete();
          if (echo == null) throw new AssertionError("Failed to deserialize");
          if (layer == echo) throw new AssertionError("Serialization did not copy");
          if (!layer.equals(echo)) throw new AssertionError("Serialization not equal");
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
  @Nonnull
  public HashMap<SerialPrecision, Layer> getModels() {
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
  @Nonnull
  public SerializationTest setPersist(boolean persist) {
    this.persist = persist;
    return this;
  }
  
  @Nonnull
  @Override
  public String toString() {
    return "SerializationTest{" +
      "models=" + models +
      ", persist=" + persist +
      '}';
  }
}
