# ls 
alias ls='ls -b --color'
alias l='ls -lh'
alias ll='ls -alh'
alias lr='ls -R'

# cp mv rm
alias cp='cp -i'
alias mv='mv -i'
alias rm='rm -i'

# grep
alias grep='grep --color=always'

# less
alias less='less -R'
alias le='/usr/share/vim/vim74/macros/less.sh'

# ptt
alias ptt='ssh bbs@ptt.cc'
alias ptt2='ssh bbs@ptt2.cc'

# screen if there is screen enter it, otherwise create new screen.
alias ssc='TERM=screen screen -d -R'
alias sc='screen -d -R'

# python
alias py='python'

# ssh
sshcml(){
	ssh -X hsinfu@cml$1.csie.ntu.edu.tw
}
sshlinuxr(){
	ssh -X r02922054@linux$1.csie.ntu.edu.tw
}
sshlinuxb(){
	ssh -X b98902038@linux$1.csie.ntu.edu.tw
}
alias sshh71='ssh -X miralab@h71.cc.ntu.edu.tw'
alias sshphi='ssh -X guest04@140.112.2.68'
alias sshsiggraph='ssh -X hsinfu@cmlsiggraph.csie.ntu.edu.tw'


alias tmux='TERM=xterm-256color tmux -2'
#alias tmux='tmux a || tmux'

# github
export HUB_DIR=/home/master/02/hsinfu/hub/
export PATH=${PATH}:${HUB_DIR}
eval "$(hub alias -s)"


# bash color
#PS1="$(if [[ ${EUID} == 0 ]]; then echo '\[\033[01;31m\]\h'; else echo '\[\033[01;32m\]\u@\h'; fi)\[\033[01;34m\] \w \$([[ \$? != 0 ]] && echo \"\[\033[01;31m\]:(\[\033[01;34m\] \")\$\[\033[00m\] "
export PS1="\[\e[1;34m\]\u@\h\[\e[m\] \[\e[1;33m\]\w\[\e[m\]\n\[\e[1;32m\]$\[\e[m\] "
export LS_COLORS="di=01;33"

export Caffe_TOOLS_DIR=/home/master/02/hsinfu/Caffe/build/tools
export MKL_DIR=/home/master/02/hsinfu/intel/mkl
CUDA_DIR=/home/master/02/shiro/cuda-7.0

#config from hsinfu
#export PATH=${PATH}:${Caffe_TOOLS_DIR}
#export LD_LIBRARY_PATH=${MKL_DIR}/lib/intel64:${LD_LIBRARY_PATH}
#config from shiro
export PATH=$CC:$CUDA_DIR/bin:/home/exta/b01902004/intel/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$MKL_DIR/lib/intel64:/home/extra/b01902004/intel/lib/intel64:/home/extra/b01902004/intel/mkl/lib/intel64/:/home/master/02/hsinfu/cuda6.5/lib:$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64/:$HOME/lib:/usr/lib/


export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
#export LC_ALL=zh_TW.UTF-8
#export LANG=zh_TW.UTF-8
#export LANGUAGE=zh_TW.UTF-8


source /etc/bash_completion
source /home/master/02/hsinfu/hub/hub.bash_completion.sh



