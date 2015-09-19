package com.simiacryptus.mindseye.training;

public class RateMonitor {
  private double counter0 = 0;
  private double counter1 = 0;
  private final double halfLifeMs;
  private long lastUpdateTime = System.currentTimeMillis();
  public final long startTime = System.currentTimeMillis();

  public RateMonitor(final double halfLifeMs) {
    super();
    this.halfLifeMs = halfLifeMs;
  }

  public double add(final double value) {
    final long prevUpdateTime = this.lastUpdateTime;
    final long now = System.currentTimeMillis();
    this.lastUpdateTime = now;
    final long elapsedMs = now - prevUpdateTime;
    final double elapsedHalflifes = elapsedMs / this.halfLifeMs;
    this.counter0 += elapsedMs;
    this.counter1 += value;
    final double v = this.counter1 / this.counter0;
    final double f = Math.pow(0.5, elapsedHalflifes);
    this.counter0 *= f;
    this.counter1 *= f;
    return v;
  }
}