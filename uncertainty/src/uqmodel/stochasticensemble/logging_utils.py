import logging
import logging.config
import os


def init_logging(logger, logfile, append=True):
    conf_file = os.path.join('logconf', 'logger.ini')
    if os.path.exists(conf_file):
        logfilename = os.path.splitext(os.path.basename(logfile))[0] + ".log"
        if not append and os.path.exists(logfilename):
            logfilename = (
                os.path.splitext(os.path.basename(logfile))[0]
                + "_"
                + str(os.getpid())
                + ".log"
            )
        logging.config.fileConfig(conf_file, defaults={"logfilename": logfilename})
    else:
        logging.basicConfig(level=logging.INFO)

    logger.debug("running from {}".format(os.getcwd()))
