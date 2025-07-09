from loguru import logger
import sys
#logger.remove()
#logger.add(sys.stdout, level='WARNING')
logger.add("my_log.log", level="DEBUG")


logger.info('We start something')
logger.debug('We debug something')
logger.warning('Be aware of this logger!')
logger.error('Something went wrong')
logger.critical('Nothing is as critical as death')