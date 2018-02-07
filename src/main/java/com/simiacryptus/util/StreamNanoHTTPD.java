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

package com.simiacryptus.util;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
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
   * The Custom handlers.
   */
  public final Map<String, Function<IHTTPSession, Response>> customHandlers = new HashMap<>();
  /**
   * The Data reciever.
   */
  @javax.annotation.Nonnull
  public final TeeOutputStream dataReciever;
  @javax.annotation.Nonnull
  private final File file;
  @javax.annotation.Nonnull
  private final URI gatewayUri;
  private final String mimeType;
  private final ExecutorService pool = Executors.newCachedThreadPool(new ThreadFactoryBuilder().setDaemon(true).build());
  
  
  /**
   * Instantiates a new Stream nano httpd.
   *
   * @param port     the port
   * @param mimeType the mime type
   * @param file     the file
   * @throws IOException the io exception
   */
  public StreamNanoHTTPD(final int port, final String mimeType, @javax.annotation.Nonnull final File file) throws IOException {
    super(port);
    this.file = file;
    this.mimeType = mimeType;
    try {
      gatewayUri = new URI(String.format("http://localhost:%s/%s", port, file.getName()));
    } catch (@javax.annotation.Nonnull final URISyntaxException e) {
      throw new RuntimeException(e);
    }
    dataReciever = new TeeOutputStream(new FileOutputStream(file), true) {
      @Override
      public void close() throws IOException {
        try {
          Thread.sleep(100);
          StreamNanoHTTPD.this.stop();
        } catch (@javax.annotation.Nonnull final Exception e) {
          e.printStackTrace();
        }
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
  public static Function<IHTTPSession, Response> asyncHandler(@javax.annotation.Nonnull final ExecutorService pool, final String mimeType, @javax.annotation.Nonnull final Consumer<OutputStream> logic, final boolean async) {
    return session -> {
      @javax.annotation.Nonnull final PipedInputStream snk = new PipedInputStream();
      @javax.annotation.Nonnull final Semaphore onComplete = new Semaphore(0);
      pool.submit(() -> {
        try (@javax.annotation.Nonnull OutputStream out = new BufferedOutputStream(new AsyncOutputStream(new PipedOutputStream(snk)))) {
          try {
            logic.accept(out);
          } finally {
            onComplete.release();
          }
        } catch (@javax.annotation.Nonnull final IOException e) {
          e.printStackTrace();
          throw new RuntimeException(e);
        }
      });
      if (!async) {
        try {
          onComplete.acquire();
        } catch (@javax.annotation.Nonnull final InterruptedException e) {
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
  @javax.annotation.Nonnull
  public static OutputStream create(final int port, @javax.annotation.Nonnull final File path, final String mimeType) throws IOException {
    return new StreamNanoHTTPD(port, mimeType, path).init().dataReciever;
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
  public static Function<IHTTPSession, Response> syncHandler(final ExecutorService pool, final String mimeType, @javax.annotation.Nonnull final Consumer<OutputStream> logic, final boolean async) {
    return session -> {
      try (@javax.annotation.Nonnull ByteArrayOutputStream out = new ByteArrayOutputStream()) {
        logic.accept(out);
        out.flush();
        final byte[] bytes = out.toByteArray();
        return NanoHTTPD.newFixedLengthResponse(Response.Status.OK, mimeType, new ByteArrayInputStream(bytes), bytes.length);
      } catch (@javax.annotation.Nonnull final IOException e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }
    };
  }
  
  /**
   * Add async handler.
   *
   * @param path     the path
   * @param mimeType the mime type
   * @param logic    the logic
   * @param async    the async
   */
  public void addAsyncHandler(final String path, final String mimeType, @javax.annotation.Nonnull final Consumer<OutputStream> logic, final boolean async) {
    addSessionHandler(path, StreamNanoHTTPD.asyncHandler(pool, mimeType, logic, async));
  }
  
  /**
   * Add session handler function.
   *
   * @param path  the path
   * @param value the value
   * @return the function
   */
  public Function<IHTTPSession, Response> addSessionHandler(final String path, final Function<IHTTPSession, Response> value) {
    return customHandlers.put(path, value);
  }
  
  /**
   * Add sync handler.
   *
   * @param path     the path
   * @param mimeType the mime type
   * @param logic    the logic
   * @param async    the async
   */
  public void addSyncHandler(final String path, final String mimeType, @javax.annotation.Nonnull final Consumer<OutputStream> logic, final boolean async) {
    addSessionHandler(path, StreamNanoHTTPD.syncHandler(pool, mimeType, logic, async));
  }
  
  /**
   * Init stream nano httpd.
   *
   * @return the stream nano httpd
   * @throws IOException the io exception
   */
  @javax.annotation.Nonnull
  public StreamNanoHTTPD init() throws IOException {
    StreamNanoHTTPD.this.start(30000);
    new Thread(() -> {
      try {
        Thread.sleep(100);
        Desktop.getDesktop().browse(gatewayUri);
      } catch (@javax.annotation.Nonnull final Exception e) {
        e.printStackTrace();
      }
    }).start();
    return this;
  }
  
  @Override
  public Response serve(final IHTTPSession session) {
    String requestPath = session.getUri();
    while (requestPath.startsWith("/")) {
      requestPath = requestPath.substring(1);
    }
    if (requestPath.equals(file.getName())) {
      try {
        @javax.annotation.Nonnull final Response response = NanoHTTPD.newChunkedResponse(Response.Status.OK, mimeType, new BufferedInputStream(dataReciever.newInputStream()));
        response.setGzipEncoding(false);
        return response;
      } catch (@javax.annotation.Nonnull final IOException e) {
        throw new RuntimeException(e);
      }
    }
    else {
      @javax.annotation.Nonnull final File file = new File(this.file.getParent(), requestPath);
      if (customHandlers.containsKey(requestPath)) {
        return customHandlers.get(requestPath).apply(session);
      }
      else if (file.exists()) {
        try {
          return NanoHTTPD.newFixedLengthResponse(Response.Status.OK, null, new FileInputStream(file), file.length());
        } catch (@javax.annotation.Nonnull final FileNotFoundException e) {
          throw new RuntimeException(e);
        }
      }
      else {
        return NanoHTTPD.newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "Not Found");
      }
    }
  }
  
  @Override
  protected boolean useGzipWhenAccepted(final Response r) {
    return false;
  }
}
