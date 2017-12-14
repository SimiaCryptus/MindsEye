# ProductLayer
## Double3
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.ProductLayer",
      "id": "58e254a9-162b-484b-8175-3b79fa7e7560",
      "isFrozen": false,
      "name": "ProductLayer/58e254a9-162b-484b-8175-3b79fa7e7560"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ -1.136 ], [ 0.884 ] ],
    	[ [ -0.464 ], [ 0.424 ] ]
    ],
    [
    	[ [ -0.748 ], [ 0.028 ] ],
    	[ [ 1.812 ], [ 0.68 ] ]
    ],
    [
    	[ [ -1.132 ], [ 1.716 ] ],
    	[ [ -0.692 ], [ -1.736 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.9618920959999998 ], [ 0.042474432 ] ],
    	[ [ 0.5818114560000001 ], [ -0.50052352 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.8467359999999999 ], [ 0.048048 ] ],
    	[ [ -1.253904 ], [ -1.18048 ] ]
    ],
    [
    	[ [ 1.2859519999999998 ], [ 1.516944 ] ],
    	[ [ 0.321088 ], [ -0.7360639999999999 ] ]
    ],
    [
    	[ [ 0.8497279999999999 ], [ 0.024752 ] ],
    	[ [ -0.8407680000000001 ], [ 0.28832 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.05 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.05 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.048 ], [ -0.028 ] ],
    	[ [ -0.276 ], [ -1.676 ] ]
    ],
    [
    	[ [ 0.108 ], [ 1.5 ] ],
    	[ [ -1.476 ], [ -1.936 ] ]
    ],
    [
    	[ [ 0.544 ], [ 1.852 ] ],
    	[ [ -0.388 ], [ -0.324 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.8016044087306795, negative=3, min=-1.676, max=-1.676, mean=-0.483, count=4.0, positive=1, stdDev=0.6991215917134873, zeros=0},
    {meanExponent=-0.08362331874949289, negative=2, min=-1.936, max=-1.936, mean=-0.45099999999999996, count=4.0, positive=2, stdDev=1.3578228897761297, zeros=0},
    {meanExponent=-0.2243458455387713, negative=2, min=-0.324, max=-0.324, mean=0.421, count=4.0, positive=2, stdDev=0.9044882531022723, zeros=0}
    Output: [
    	[ [ 0.0028200960000000002 ], [ -0.077784 ] ],
    	[ [ -0.158061888 ], [ -1.051294464 ] ]
    ]
    Outputs Statistics: {meanExponent=-1.1095735730189438, negative=3, min=-1.051294464, max=-1.051294464, mean=-0.321080064, count=4.0, positive=1, stdDev=0.4254092982292596, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.048 ], [ -0.028 ] ],
    	[ [ -0.276 ], [ -1.676
```
...[skipping 3284 bytes](etc/80.txt)...
```
    itive=3, stdDev=0.7857782752609033, zeros=12}
    Measured Feedback: [ [ 0.005183999999998357, 0.0, 0.0, 0.0 ], [ 0.0, 0.40737600000007035, 0.0, 0.0 ], [ 0.0, 0.0, -0.04199999999995874, 0.0 ], [ 0.0, 0.0, 0.0, 3.244735999998749 ] ]
    Measured Statistics: {meanExponent=-0.8852277274803365, negative=1, min=3.244735999998749, max=3.244735999998749, mean=0.22595599999992869, count=16.0, positive=3, stdDev=0.7857782752606032, zeros=12}
    Feedback Error: [ [ -1.6436504934880247E-15, 0.0, 0.0, 0.0 ], [ 0.0, 7.033262861000367E-14, 0.0, 0.0 ], [ 0.0, 0.0, 4.126560204653629E-14, 0.0 ], [ 0.0, 0.0, 0.0, -1.2505552149377763E-12 ] ]
    Error Statistics: {meanExponent=-13.306085654811735, negative=2, min=-1.2505552149377763E-12, max=-1.2505552149377763E-12, mean=-7.128753967342027E-14, count=16.0, positive=2, stdDev=3.0508502531583686E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.6680e-14 +- 2.2225e-13 [0.0000e+00 - 1.2506e-12] (48#)
    relativeTol: 2.7596e-13 +- 2.7686e-13 [1.5426e-14 - 8.8304e-13] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.6680e-14 +- 2.2225e-13 [0.0000e+00 - 1.2506e-12] (48#), relativeTol=2.7596e-13 +- 2.7686e-13 [1.5426e-14 - 8.8304e-13] (12#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[2, 2, 1]
    	[2, 2, 1]
    	[2, 2, 1]
    Performance:
    	Evaluation performance: 0.000545s +- 0.000046s [0.000485s - 0.000623s]
    	Learning performance: 0.000223s +- 0.000020s [0.000187s - 0.000244s]
    
```

