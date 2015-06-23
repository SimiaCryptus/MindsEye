package com.simiacryptus.mindseye;

import java.awt.image.BufferedImage;

public class LabeledImage {
  public final BufferedImage img;
  public final String name;
  public LabeledImage(BufferedImage img, String name) {
    super();
    this.img = img;
    this.name = name;
  }
  
}