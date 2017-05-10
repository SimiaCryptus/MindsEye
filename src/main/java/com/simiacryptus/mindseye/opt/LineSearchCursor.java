package com.simiacryptus.mindseye.opt;

/**
 * Created by Andrew Charneski on 5/9/2017.
 */
public interface LineSearchCursor {

    LineSearchPoint step(double alpha, TrainingMonitor monitor);
}
