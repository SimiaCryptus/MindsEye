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

package com.simiacryptus.util;

import com.simiacryptus.util.io.AsyncOutputStream;
import com.simiacryptus.util.io.TeeOutputStream;
import fi.iki.elonen.NanoHTTPD;

import java.awt.*;
import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * The type Stream nano httpd.
 */
public class StreamNanoHTTPD extends NanoHTTPD {
  /**
   * The Data reciever.
   */
  public final TeeOutputStream dataReciever;
  /**
   * The Custom handlers.
   */
  public final Map<String, Function<IHTTPSession, Response>> customHandlers = new HashMap<>();
  private final String mimeType;
  private final URI gatewayUri;
  private final File file;
  private final ExecutorService pool = Executors.newCachedThreadPool();
  
  
  /**
   * Instantiates a new Stream nano httpd.
   *
   * @param port     the port
   * @param mimeType the mime type
   * @param file     the file
   * @throws IOException the io exception
   */
  public StreamNanoHTTPD(int port, String mimeType, File file) throws IOException {
    super(port);
    this.file = file;
    this.mimeType = mimeType;
    try {
      this.gatewayUri = new URI(String.format("http://localhost:%s/%s", port, file.getName()));
    } catch (URISyntaxException e) {
      throw new RuntimeException(e);
    }
    this.dataReciever = new TeeOutputStream(new FileOutputStream(file), true) {
      @Override
      public void close() throws IOException {
        try {
          Thread.sleep(100);
          StreamNanoHTTPD.this.stop();
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    };
  }
  
  /**
   * Sync handler function.
   *
   * @param pool     the pool
   * @param mimeType the mime type
   * @param logic    the logic
   * @param async    the async
   * @return the function
   */
  public static Function<IHTTPSession, Response> syncHandler(ExecutorService pool, String mimeType, Consumer<OutputStream> logic, boolean async) {
    return session -> {
      try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
        logic.accept(out);
        out.flush();
        byte[] bytes = out.toByteArray();
        return NanoHTTPD.newFixedLengthResponse(Response.Status.OK, mimeType, new ByteArrayInputStream(bytes), bytes.length);
      } catch (IOException e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }
    };
  }
  
  /**
   * Async handler function.
   *
   * @param pool     the pool
   * @param mimeType the mime type
   * @param logic    the logic
   * @param async    the async
   * @return the function
   */
  public static Function<IHTTPSession, Response> asyncHandler(ExecutorService pool, String mimeType, Consumer<OutputStream> logic, boolean async) {
    return session -> {
      PipedInputStream snk = new PipedInputStream();
      Semaphore onComplete = new Semaphore(0);
      pool.submit(() -> {
        try (OutputStream out = new BufferedOutputStream(new AsyncOutputStream(new PipedOutputStream(snk)))) {
          try {
            logic.accept(out);
          } finally {
            onComplete.release();
          }
        } catch (IOException e) {
          e.printStackTrace();
          throw new RuntimeException(e);
        }
      });
      if (!async) {
        try {
          onComplete.acquire();
        } catch (InterruptedException e) {
          throw new RuntimeException(e);
        }
      }
      return NanoHTTPD.newChunkedResponse(Response.Status.OK, mimeType, new BufferedInputStream(snk));
    };
  }
  
  /**
   * Create output stream.
   *
   * @param port     the port
   * @param path     the path
   * @param mimeType the mime type
   * @return the output stream
   * @throws IOException the io exception
   */
  public static OutputStream create(int port, File path, String mimeType) throws IOException {
    return new StreamNanoHTTPD(port, mimeType, path).init().dataReciever;
  }
  
  /**
   * Add sync handler.
   *
   * @param path     the path
   * @param mimeType the mime type
   * @param logic    the logic
   * @param async    the async
   */
  public void addSyncHandler(String path, String mimeType, Consumer<OutputStream> logic, boolean async) {
    addSessionHandler(path, syncHandler(pool, mimeType, logic, async));
  }
  
  /**
   * Add async handler.
   *
   * @param path     the path
   * @param mimeType the mime type
   * @param logic    the logic
   * @param async    the async
   */
  public void addAsyncHandler(String path, String mimeType, Consumer<OutputStream> logic, boolean async) {
    addSessionHandler(path, asyncHandler(pool, mimeType, logic, async));
  }
  
  /**
   * Add session handler function.
   *
   * @param path  the path
   * @param value the value
   * @return the function
   */
  public Function<IHTTPSession, Response> addSessionHandler(String path, Function<IHTTPSession, Response> value) {
    return customHandlers.put(path, value);
  }
  
  /**
   * Init stream nano httpd.
   *
   * @return the stream nano httpd
   * @throws IOException the io exception
   */
  public StreamNanoHTTPD init() throws IOException {
    StreamNanoHTTPD.this.start(30000);
    new Thread(() -> {
      try {
        Thread.sleep(100);
        Desktop.getDesktop().browse(gatewayUri);
      } catch (Exception e) {
        e.printStackTrace();
      }
    }).start();
    return this;
  }
  
  @Override
  public Response serve(IHTTPSession session) {
    String requestPath = session.getUri();
    while (requestPath.startsWith("/")) requestPath = requestPath.substring(1);
    if (requestPath.equals(file.getName())) {
      try {
        Response response = NanoHTTPD.newChunkedResponse(Response.Status.OK, mimeType, new BufferedInputStream(dataReciever.newInputStream()));
        response.setGzipEncoding(false);
        return response;
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    else {
      File file = new File(this.file.getParent(), requestPath);
      if (customHandlers.containsKey(requestPath)) {
        return customHandlers.get(requestPath).apply(session);
      }
      else if (file.exists()) {
        try {
          return NanoHTTPD.newFixedLengthResponse(Response.Status.OK, null, new FileInputStream(file), file.length());
        } catch (FileNotFoundException e) {
          throw new RuntimeException(e);
        }
      }
      else {
        return NanoHTTPD.newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "Not Found");
      }
    }
  }
  
  @Override
  protected boolean useGzipWhenAccepted(Response r) {
    return false;
  }
}
