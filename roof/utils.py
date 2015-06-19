import sys
import getopt


def command_line_process():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:n:l:p:r:a:e:")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2) 
    test_percent=0.2
    non_roofs=1
    preloaded=False
    num_layers=0 #logistic
    roofs_only=True
    plot=True
    net_name=None
    epoch=250
    for opt, arg in opts:
        if opt == '-t':
            test_percent=float(arg)
        elif opt == '-n':
            non_roofs=int(float(arg))
        elif opt=='-p':
            preloaded=bool(arg)
        elif opt=='-l':
            num_layers=int(float(arg))
        elif opt=='-r':
            roofs_only=True
        elif opt=='-a':
            net_name=arg
        elif opt=='-e':
            epoch=int(float(arg))
    return test_percent, non_roofs, preloaded, num_layers, roofs_only, plot, net_name, epoch


