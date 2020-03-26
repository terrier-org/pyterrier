package org.terrier.python;


import org.slf4j.ILoggerFactory;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;
import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;

public class PTUtils
{
    /** Change the logback logging level.
     * @param logLevel: can be one of INFO, DEBUG, TRACE, WARN, ERROR. The latter is the quietest.
     * @param packageName: this is the package name. Set to null for the ROOT (default) logger.
     * @return true if logger could be adapted, false otherwise
     */
    // Based on https://examples.javacodegeeks.com/enterprise-java/logback/logback-change-log-level-runtime-example/
    public static boolean setLogLevel(String logLevel, String packageName) {
        ILoggerFactory ilf = LoggerFactory.getILoggerFactory();
        if (! (ilf instanceof LoggerContext))
        {
            //its not slf4j that is in use
            return false;
        }
        LoggerContext loggerContext = (LoggerContext) ilf;

        //see http://www.slf4j.org/apidocs/org/slf4j/ILoggerFactory.html
        if (packageName == null)
            packageName = Logger.ROOT_LOGGER_NAME;
        ch.qos.logback.classic.Logger logger = loggerContext.getLogger(packageName);
        logger.setLevel(Level.toLevel(logLevel));
        return true;
    }

}