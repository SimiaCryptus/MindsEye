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

package com.simiacryptus.util.io;

//  Originally Created by Elliot Kroo on 2009-04-25.
//
// This work is licensed under the Creative Commons Attribution 3.0 Unported
// License. To view a copy of this license, visit
// http://creativecommons.org/licenses/by/3.0/ or send a letter to Creative
// Commons, 171 Second Street, Suite 300, San Francisco, California, 94105, USA.

import javax.imageio.*;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.stream.FileImageOutputStream;
import javax.imageio.stream.ImageOutputStream;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;

/**
 * The type Gif sequence writer.
 */
public class GifSequenceWriter {
  
  /**
   * The Gif writer.
   */
  protected ImageWriter gifWriter;
  /**
   * The Image getCudaPtr param.
   */
  protected ImageWriteParam imageWriteParam;
  /**
   * The Image meta data.
   */
  protected IIOMetadata imageMetaData;
  
  /**
   * Creates a new GifSequenceWriter
   *
   * @param outputStream        the ImageOutputStream to be written to
   * @param imageType           one of the imageTypes specified in BufferedImage
   * @param timeBetweenFramesMS the time between frames in miliseconds
   * @param loopContinuously    wether the gif should loop repeatedly
   * @throws IOException the io exception
   * @author Elliot Kroo (elliot[at]kroo[dot]net)
   */
  public GifSequenceWriter(
    ImageOutputStream outputStream,
    int imageType,
    int timeBetweenFramesMS,
    boolean loopContinuously) throws IOException {
  
    gifWriter = getWriter("gif");
    imageWriteParam = gifWriter.getDefaultWriteParam();
    ImageTypeSpecifier imageTypeSpecifier = ImageTypeSpecifier.createFromBufferedImageType(imageType);
    imageMetaData = gifWriter.getDefaultImageMetadata(imageTypeSpecifier, imageWriteParam);
    String metaFormatName = imageMetaData.getNativeMetadataFormatName();
    @javax.annotation.Nonnull IIOMetadataNode root = (IIOMetadataNode) imageMetaData.getAsTree(metaFormatName);
    @javax.annotation.Nonnull IIOMetadataNode graphicsControlExtensionNode = getNode(root, "GraphicControlExtension");
    graphicsControlExtensionNode.setAttribute("disposalMethod", "none");
    graphicsControlExtensionNode.setAttribute("userInputFlag", "FALSE");
    graphicsControlExtensionNode.setAttribute("transparentColorFlag", "FALSE");
    graphicsControlExtensionNode.setAttribute("delayTime", Integer.toString(timeBetweenFramesMS / 10));
    graphicsControlExtensionNode.setAttribute("transparentColorIndex", "0");
    @javax.annotation.Nonnull IIOMetadataNode commentsNode = getNode(root, "CommentExtensions");
    commentsNode.setAttribute("CommentExtension", "Created by MindsEye");
    @javax.annotation.Nonnull IIOMetadataNode appEntensionsNode = getNode(root, "ApplicationExtensions");
    @javax.annotation.Nonnull IIOMetadataNode child = new IIOMetadataNode("ApplicationExtension");
    child.setAttribute("applicationID", "NETSCAPE");
    child.setAttribute("authenticationCode", "2.0");
    
    int loop = loopContinuously ? 0 : 1;
    child.setUserObject(new byte[]{0x1, (byte) (loop & 0xFF), (byte) ((loop >> 8) & 0xFF)});
    appEntensionsNode.appendChild(child);
    imageMetaData.setFromTree(metaFormatName, root);
    gifWriter.setOutput(outputStream);
    gifWriter.prepareWriteSequence(null);
  }
  
  /**
   * Write.
   *
   * @param gif                 the gif
   * @param timeBetweenFramesMS the time between frames ms
   * @param loopContinuously    the loop continuously
   * @param images              the images
   * @throws IOException the io exception
   */
  public static void write(File gif, int timeBetweenFramesMS, boolean loopContinuously, @javax.annotation.Nonnull BufferedImage... images) throws IOException {
    @javax.annotation.Nonnull ImageOutputStream output = new FileImageOutputStream(gif);
    try {
      @javax.annotation.Nonnull GifSequenceWriter writer = new GifSequenceWriter(output, images[0].getType(), timeBetweenFramesMS, loopContinuously);
      for (@javax.annotation.Nonnull BufferedImage image : images) {
        writer.writeToSequence(image);
      }
      writer.close();
    } finally {
      output.close();
    }
  }
  
  /**
   * Returns the first available GIF ImageWriter using ImageIO.getImageWritersBySuffix("gif").
   *
   * @param format
   * @return a GIF ImageWriter object
   * @throws IIOException if no GIF image writers are returned
   */
  private static ImageWriter getWriter(@javax.annotation.Nonnull String format) throws IIOException {
    Iterator<ImageWriter> iter = ImageIO.getImageWritersBySuffix(format);
    if (!iter.hasNext()) {
      throw new IIOException("No GIF Image Writers Exist");
    }
    else {
      return iter.next();
    }
  }
  
  /**
   * Returns an existing child node, or creates and returns a new child node (if the requested node does not exist).
   *
   * @param rootNode the <tt>IIOMetadataNode</tt> to search for the child node.
   * @param nodeName the name of the child node.
   * @return the child node, if found or a new node created with the given name.
   */
  @javax.annotation.Nonnull
  private static IIOMetadataNode getNode(
    @javax.annotation.Nonnull IIOMetadataNode rootNode,
    String nodeName) {
    int nNodes = rootNode.getLength();
    for (int i = 0; i < nNodes; i++) {
      if (rootNode.item(i).getNodeName().compareToIgnoreCase(nodeName)
        == 0) {
        IIOMetadataNode item = (IIOMetadataNode) rootNode.item(i);
        if (null == item) throw new IllegalStateException();
        return item;
      }
    }
    @javax.annotation.Nonnull IIOMetadataNode node = new IIOMetadataNode(nodeName);
    rootNode.appendChild(node);
    return (node);
  }
  
  /**
   * Write to sequence.
   *
   * @param img the img
   * @throws IOException the io exception
   */
  public void writeToSequence(@javax.annotation.Nonnull RenderedImage img) throws IOException {
    gifWriter.writeToSequence(
      new IIOImage(
        img,
        null,
        imageMetaData),
      imageWriteParam);
  }
  
  /**
   * Close this GifSequenceWriter object. This does not close the underlying stream, just finishes off the GIF.
   *
   * @throws IOException the io exception
   */
  public void close() throws IOException {
    gifWriter.endWriteSequence();
  }
}