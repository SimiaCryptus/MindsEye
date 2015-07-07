  appender("CONSOLE", ConsoleAppender) {
    encoder(PatternLayoutEncoder) {
      pattern = "\\(%F:%line\\) %msg%n"
    }
  }

appender("FILE", RollingFileAppender) {
  encoder(PatternLayoutEncoder) {
      pattern = "%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n"
  }
  rollingPolicy(TimeBasedRollingPolicy) {
    fileNamePattern = "logs/log-%d{HH-mm}.log.zip"
    timeBasedFileNamingAndTriggeringPolicy(SizeAndTimeBasedFNATP) {
	    maxFileSize = "50MB"
    }
  }
}
  root(DEBUG, ["CONSOLE", "FILE"])
  