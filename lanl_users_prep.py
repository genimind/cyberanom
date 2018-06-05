import os
import math

redevents = set()
max_len = 120


# user_names = ['U12', 'U13', 'U24', 'U66', 'U78', 'U207', 'U293', 'U453', 'U679', 'U1289', 'U1480']
# for simple testing, we'll initially ignore U66, it has 11M events
user_names_small = ['U8170', 'U3277', 'U8840', 'U7311', 'U1467', 'U1789', 'U8168', 'U1581', 'U7004', 'U9763']
user_names_moderate = ['U5254', 'U9407', 'U1592', 'U1723', 'U1106', 'U3406', 'U342', 'U1653', 
                'U20', 'U250', 'U1450', 'U1164', 'U86']
user_names_most_active = ['U12', 'U13', 'U24', 'U78', 'U207', 'U293', 'U453', 'U679', 'U1289', 'U1480']
user_names = user_names_small
users_indir = 'data/users'
users_outdir = 'data/users_feats'
redteam_fname = 'data/redteam.txt'

def transform_line(line):
    '''
        log line: line to be transformed, 
        
        replace ',' with '|' for easy of processing of each sentence later during training.
    '''
    return line.replace(',', '|')

def process_user(infile_name, outfile_name):
    '''
        process each user's file to produce a comma serparated file with the format
        "sec, day, red, sent_len, sentence" which is used as an input to the model
    '''
    with open(infile_name, 'r') as infile, open(outfile_name, 'w') as outfile:
        outfile.write('sec,day,red,seq_len,sentence\n') # header
        redcount = 0
        for line in infile.readlines():
            line = line.strip().split(',')
            sentence = ','.join(line[1:])
            diff = max_len - len(sentence)
            sec = line[0]
            day = math.floor(int(sec)/86400)
            red = 0
            redentry = "{0},{1},{2},{3}".format(line[0], line[1], line[3], line[4])
            red += int(redentry in redevents) # 1 if line is red event
            redcount += red
            translated = transform_line(sentence)
            outfile.write("%s,%s,%s,%s,%s\n" % (sec, day, 
                                                    red, len(sentence), translated))
        print('done - red team events:', redcount)
        outfile.close()
        infile.close()


def split_to_multiple_days(user_infile, user_outdir):
    '''
        Split the user data to multiple files, one for each day, and stored in a 
        directory with the user name. The files are used as an input to the model
    '''
    with open(user_infile, 'r') as data:
        current_day = 0
        outfile = open(user_outdir + str(current_day) + '.txt', 'w')
        print('processing:', u, '...', current_day, end='')
        data.readline()
        for line in data.readlines():
            try:
                line_items = line.strip().split(',')
                day = int(line_items[1])
                if day == current_day:
                    outfile.write(line)
                else:
                    outfile.close()
                    current_day = day
                    outfile = open(user_outdir + str(current_day) + '.txt', 'w')
                    print(',', current_day, end='')
                    outfile.write(line)
            except:
                print('error processing file.... line: ', line)
        outfile.close()
        print(' ...Done!')

if __name__ == "__main__":
    # make sure we have the output dir
    if not os.path.exists(users_outdir):
        os.makedirs(users_outdir)

    with open(redteam_fname, 'r') as red:
        for line in red:
            redevents.add(line.strip())

    for u in user_names:
        user_infile = '{0}/{1}.txt'.format(users_indir, u)
        user_outfile = '{0}/{1}_feats.txt'.format(users_outdir, u)
        print('processing: ', u, '...', end='')
        process_user(user_infile, user_outfile)


    # The final preprocessing step is to split the translated data into multiple files; one for each day.

    for u in user_names:
        user_infile = '{0}/{1}_feats.txt'.format(users_outdir, u)
        user_outdir = '{0}/{1}/'.format(users_outdir, u)

        if not os.path.exists(user_outdir):
            os.makedirs(user_outdir)
        
        split_to_multiple_days(user_infile, user_outdir)


