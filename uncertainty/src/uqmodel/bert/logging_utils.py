"""
https://github.com/rob-blackbourn/medium-queue-logging/.
"""
import atexit
import logging
import os
import yaml
import logging.config
import logging.handlers
from queue import Queue

log_config_filepath = os.path.join('logconf', 'logger_mp_bertps.yml')
GLOBAL_LOG_FILEPATH = None

def _resolve_handlers(lh):
    if not isinstance(lh, logging.config.ConvertingList):
        return lh

    # Indexing the list performs the evaluation.
    return [lh[i] for i in range(len(lh))]


def _resolve_queue(q):
    if not isinstance(q, logging.config.ConvertingDict):
        return q
    if '__resolved_value__' in q:
        return q['__resolved_value__']

    cname = q.pop('class')
    klass = q.configurator.resolve(cname)
    props = q.pop('.', None)
    kwargs = {k: q[k] for k in q if logging.config.valid_ident(k)}
    result = klass(**kwargs)
    if props:
        for name, value in props.items():
            setattr(result, name, value)

    q['__resolved_value__'] = result
    return result


class QueueListenerHandler(logging.handlers.QueueHandler):

    def __init__(self, handlers, respect_handler_level=False, auto_run=True, queue=None):
        if not queue:
            queue = Queue(-1)
        queue = _resolve_queue(queue)
        super().__init__(queue)
        handlers = _resolve_handlers(handlers)
        self._listener = logging.handlers.QueueListener(
            self.queue,
            *handlers,
            respect_handler_level=respect_handler_level)
        if auto_run:
            self.start()
            atexit.register(self.stop)

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()

    def emit(self, record):
        return super().emit(record)

def init_logging(logfile, append=True, config_file:str=log_config_filepath):
    global GLOBAL_LOG_FILEPATH
    
    if os.path.exists(config_file):
        logfilename = os.path.splitext(os.path.basename(logfile))[0] + '.log'
        if not append and os.path.exists(logfilename):
            logfilename = os.path.splitext(os.path.basename(logfile))[0] + '_' + str(os.getpid()) + '.log'
        with open(config_file, 'rt') as f:
            logging_config = yaml.load(f, Loader=yaml.FullLoader) # nosec
        logging_config['handlers']['file']['filename'] = logfilename
        GLOBAL_LOG_FILEPATH = logfilename
        logging.config.dictConfig(logging_config)
    else:
        logging.warning('log configuration file {} inaccessible, use basic configuration'
                        .format(log_config_filepath))
        logging.basicConfig(level=logging.INFO)
    
def get_global_logfilename():
    global GLOBAL_LOG_FILEPATH
    return GLOBAL_LOG_FILEPATH