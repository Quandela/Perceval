
#header#{{{#
scriptPath="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
advancedPath="${scriptPath}/.."
#}}}#

##                                                                                                ##
## arguments                                                                                      ##
##                                                                                                ##

modes=""
nphotons="4"
backends="SLOS SLAP SLOS_CPP SLOS_V2 SLOS_V3"
mask=""
compute_method=1

# message strings
USAGESTR="Usage: $0 [--modes/-m] 20 [--nphotons/-n] 10 [--backends/-b] \"SLOS_V2 SLOS_V3\" [--mask/-k] \"1010**\" [--compute/-c] 2"
ERRORSTR="Failed to parse options... exiting."

# option strings
SHORTOPTIONS="m:n:i:b:k:c:"
LONGOPTIONS="modes:,nphotons:,minphotons:,backends:,mask:,compute:"

# read the options
ARGS=$(getopt --options $SHORTOPTIONS --longoptions $LONGOPTIONS --name "$0" -- "$@") || exit 1
eval "set -- $ARGS"

# extract options and their arguments into variables
while true ; do
  case "$1" in
    -h | --help ) echo "$USAGESTR" ; exit 0 ;;
    -m | --modes ) modes="$2" ; shift 2 ;;
    -n | --nphotons ) nphotons="$2" ; shift 2 ;;
    -b | --backends ) backends="$2" ; shift 2 ;;
    -k | --mask ) mask="$2" ; shift 2 ;;
    -c | --compute ) compute="$2" ; shift 2 ;;
    -- ) shift ; break ;;
    *) echo "$ERRORSTR" >&2 ; exit 1 ;;
  esac
done

for backend in $backends
do
    python ${scriptPath}/Bench_backends.py -u -m ${modes} -n ${nphotons} -b "${backend}" -c ${compute} -k "$mask"
done
