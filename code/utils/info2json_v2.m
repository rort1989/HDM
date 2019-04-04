function str = info2json_v2(cameraID,personID,x,y,z,nx,ny,nz,gestureType,direction,factor)
% form the string for publishing results in json file format

dq = '"';
bkf = '{';
bkb = '}';
cma = ',';
cl = ':';

c = clock;
str = strcat(bkf,dq,'timeStamp',dq,cl);
timestamp = sprintf('%d-%02i-%d %d.%02i.%02i.%02i',c(1),c(2),c(3),c(4),c(5),floor(c(6)),round(100*(c(6)-floor(c(6)))));
str = strcat(str,dq,timestamp,dq,cma);

str = strcat(str,dq,'cameraID',dq,cl,dq,cameraID,dq,cma);

str = strcat(str,dq,'personID',dq,cl,sprintf('%d',personID),cma);

str = strcat(str,dq,'bodyPose',dq,cl,bkf);
str = strcat(str,dq,'X',dq,cl,sprintf('%.2f',x),cma);
str = strcat(str,dq,'Y',dq,cl,sprintf('%.2f',y),cma);
str = strcat(str,dq,'Z',dq,cl,sprintf('%.2f',z),cma);
str = strcat(str,dq,'pitch',dq,cl,sprintf('%.4f',nx),cma);
str = strcat(str,dq,'yaw',dq,cl,sprintf('%.4f',ny),cma);
str = strcat(str,dq,'roll',dq,cl,sprintf('%.4f',nz));
str = strcat(str,bkb,cma);

str = strcat(str,dq,'gestureType',dq,cl,dq,gestureType,dq,cma);

str = strcat(str,dq,'parameters',dq,cl,bkf);
str = strcat(str,dq,'direction',dq,cl,dq,direction,dq,cma);
% str = strcat(str,dq,'direction',dq,cl,bkf);
% str = strcat(str,dq,'X',dq,cl,sprintf('%.2f',direction(1)),cma);
% str = strcat(str,dq,'Y',dq,cl,sprintf('%.2f',direction(2)),cma);
% str = strcat(str,dq,'Z',dq,cl,sprintf('%.2f',direction(3)));
% str = strcat(str,bkb,cma);

str = strcat(str,dq,'factor',dq,cl,sprintf('%.1f',factor));

str = strcat(str,bkb,bkb);