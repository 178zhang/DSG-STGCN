function han = showsurface3(vertices, faces, para, varargin)

vc.vc = vertices;
vc.tri = faces;

if nargin<3
    para.normalize=1;
else
    if ~isfield(para,'normalize')
           para.normalize=1;
    end
end

myviewdir=[1 0 .5];
numsubplots=1;
mymarkersize=10;
mylinewidth=40;
myfontsize = 45;
mydirfontsize = 30;
fsv=0;
fsv2=10;
dipcolorstyle='uni';
dipscal=20;
dipnames = 0;
showdirections = 0;
directions = 0;
dipcolorbars = 1;
dipcolors={'k','b','r','y','g','c','m','y'};
if nargin>2 %% ==============
   if isfield(para,'myviewdir');
     myviewdir=para.myviewdir;
   end   
  if isfield(para,'numsubplots');
     numsubplots=para.numsubplots;
  end  
  if isfield(para,'mymarkersize');
     mymarkersize=para.mymarkersize;
  end  
  if isfield(para,'mylinewidth');
     mylinewidth=para.mylinewidth;
  end
  if isfield(para,'myfontsize');
     myfontsize=para.myfontsize;
  end
  if isfield(para,'mydirfontsize');
     mydirfontsize=para.mydirfontsize;
  end
  if isfield(para,'fsv');
     fsv=para.fsv;
  end
 if isfield(para,'dipcolorstyle');
    dipcolorstyle=para.dipcolorstyle;
 end
  if isfield(para,'dipscal');
     dipscal=para.dipscal;
  end
  if isfield(para,'dipnames');
     dipnames=para.dipnames;
  end
  if isfield(para,'showdirections');
     showdirections=para.showdirections;
  end
   if isfield(para,'directions');
     directions=para.directions;
  end
   if isfield(para,'dipcolorbars')
     dipcolorbars=para.dipcolorbars;
   end
   if isfield(para,'dipcolors')
     dipcolors=para.dipcolors;
   end
end
   

if isfield(vc,'vc')
    loc=vc.vc(:,1:3);
elseif isfield(vc,'vc_ori')
    loc=vc.vc_ori(:,1:3);
elseif isfield(vc,'vc_ori_model')
    loc=vc.vc_ori_model(:,1:3);
elseif isfield(vc,'vc_coarse') 
    loc=vc.vc_coarse(:,1:3);
else
    error('first argument structure should be a structure with field vc_ori_model or vc_ori or vc or vc_coarse (list of surface points)')
end

if isfield(vc,'tri');
    tri=vc.tri(:,1:3);
elseif isfield(vc,'tri_coarse')
    disp('using coarse model to show vc');
    loc=vc.vc_coarse(:,1:3);    
    tri=vc.tri_coarse(:,1:3);
else
    error('first argument should be a structure with field named tri or tri_coarse (list of triangles)')
end


if isfield(vc,'faceori');
  para.faceori=vc.faceori;
else
  pp.vc=loc;
  pp.tri=tri;
  %pp=vc2vcnormals(pp); para.faceori=pp.faceori; para.vertexori=pp.vertexori;
end
if isfield(vc,'vertexori');
  para.vertexori=vc.vertexori;
elseif isfield(para,'vertexori')
else  
  pp.vc=loc;
  pp.tri=tri;
  %pp=vc2vcnormals(pp); para.faceori=pp.faceori; para.vertexori=pp.vertexori;
end

ndum=0;  nval=0;ndotvec=0;
if nargin>3 %% ==============
    nss=length(varargin);
      for k=1:nss
        source_x=varargin{k};
        [ns,ndum]=size(source_x);
         if ndum==3 || ndum==6 || ndum==4;
            ndotvec=ndotvec+1;  
            source{ndotvec}=source_x;
         elseif ndum==1 || ndum==2
            nval=nval+1;  
            source_val{nval}=source_x(:, 1);
            if nval==1 && length(find(source_x(:, 1))) > 0;
                para.voxelfield=source_x(:, 1);
            end  
            if ndum == 2
                alpha_val{nval}=source_x(:, 2);
                if nval==1;
                    para.alphafield=source_x(:, 2);
                end
            end
         end
     end
end

%  keyboard

colors_x = dipcolors;
w=[1,1,.999];
for i=1:ndotvec
    colors{i}=colors_x{mod(i-1,6)+1};
end
if strcmp(dipcolorstyle,'mixed');
    npall=zeroth(ndotvec,1)
    for k=1:ndotvec;
        [npall(k),ndum]=size(source{k});
    end
    npmax=max(npall);
    for i=1:npmax
        colors{i}=colors_x{mod(i,6)+1};
    end
end

if isfield(para,'dotveccolors');
    nc=length(para.dotveccolors);
    for i=1:nc
        colors{i}=para.dotveccolors{i};
    end
end


figscale=1.1;
mins=min(loc);
maxs=max(loc);
figscalevec=figscale*[mins(1) maxs(1) mins(2) maxs(2) mins(3) maxs(3)];

if showdirections
  h(1) = text(mins(1)-1, (maxs(2)+mins(2))/2, (maxs(3)+mins(3))/2, 'L', 'fontsize', mydirfontsize, 'fontweight', 'bold', 'color', 'b');
  set(h(1), 'horizontalalignment', 'center', 'verticalalignment', 'middle')
  h(2) = text(maxs(1)+1, (maxs(2)+mins(2))/2, (maxs(3)+mins(3))/2, 'R', 'fontsize', mydirfontsize, 'fontweight', 'bold', 'color', 'b');
  set(h(2), 'horizontalalignment', 'center', 'verticalalignment', 'middle')
  h(3) = text((maxs(1)+mins(1))/2, mins(2)-1, (maxs(3)+mins(3))/2, 'P', 'fontsize', mydirfontsize, 'fontweight', 'bold', 'color', 'b');
  set(h(3), 'horizontalalignment', 'center', 'verticalalignment', 'middle')
  h(4) = text((maxs(1)+mins(1))/2, maxs(2)+1, (maxs(3)+mins(3))/2, 'A', 'fontsize', mydirfontsize, 'fontweight', 'bold', 'color', 'b');
  set(h(4), 'horizontalalignment', 'center', 'verticalalignment', 'middle')
  h(5) = text((maxs(1)+mins(1))/2, (maxs(2)+mins(2))/2, mins(3)-0.5, 'I', 'fontsize', mydirfontsize, 'fontweight', 'bold', 'color', 'b');
  set(h(5), 'horizontalalignment', 'center', 'verticalalignment', 'middle')
  h(6) = text((maxs(1)+mins(1))/2, (maxs(2)+mins(2))/2, maxs(3)+0.5, 'S', 'fontsize', mydirfontsize, 'fontweight', 'bold', 'color', 'b');
  set(h(6), 'horizontalalignment', 'center', 'verticalalignment', 'middle')
end
% keyboard
han = showvc_prog_plain(loc,tri,myviewdir,para);
hold on;

if showdirections
  cp = get(gca, 'cameraposition');
  ht = [get(h, 'position')];
  ht = reshape([ht{:}], 3, 6);
  if length(directions) == 6
      set(h(find(~directions)), 'visible', 'off')    
  else
      md = eucl([ht cp']');
      [so in] = sort(md(1:end-1, end), 'ascend');
      set(h(in([1 end])), 'visible', 'off')
  end
end

for k=1:ndotvec;
    source_x=source{k}; 
    [np,ndum]=size(source_x);
    if ndum==3
        for i=1:np;
          if strcmp(dipcolorstyle,'mixed')
              colorloc=colors{i};
          else
              colorloc=colors{k};
          end
          points_loc=source_x(i,1:3)+fsv*myviewdir;
           [xs,ys,zs] = sphere;
           hs = surf((xs*mymarkersize/3)+points_loc(1, 1),(ys*mymarkersize/3)+points_loc(1, 2),(zs*mymarkersize/3)+points_loc(1, 3));
           set(hs, 'facecolor', colorloc, 'edgecolor', 'none')

          if isequal(dipnames, 1)
              points_loc=source_x(i,1:3)+fsv2*myviewdir;
              ht = text(points_loc(1, 1)+15, points_loc(1, 2)-15, points_loc(1, 3)+30, num2str(i), 'fontsize', myfontsize, 'fontweight', 'bold');
              set(ht, 'color', colorloc, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle')
          elseif iscell(dipnames)
              points_loc=source_x(i,1:3)+fsv2*myviewdir;
              ht = text(points_loc(1, 1)+15, points_loc(1, 2)-15, points_loc(1, 3)+30, dipnames{k}{i}, 'fontsize', myfontsize, 'fontweight', 'bold');
              set(ht, 'color', colorloc, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle')
          end
        end
%        plot3(source_x(:,1),source_x(:,2),source_x(:,3),'.','markersize',mymarkersize,'color',mycolor);
    elseif ndum==4 
      

%       voxelfield=source_x;
%       voxelkont=1;
      if isfield(para,'colorlimits');
        colmin=para.colorlimits(1);
        colmax=para.colorlimits(2);
      end
      
      source_val = source_x(:, 4);
      source_val_max=max(abs(source_val));
      source_val_min=min(abs(source_val));
      
      if isfield(para, 'climPolicy')
        climPolicy = para.climPolicy;
      else
        if ~isfield(para, 'colorlimits')
          if sign(max(source_val)) == sign(min(source_val)) || abs(min(source_val)) < eps
            climPolicy = 'minmax';
          else
            climPolicy = 'sym';
          end
        else
          climPolicy = 'none';
        end
      end
      
      if isequal(climPolicy, 'minmax')
        colmax=max(source_val);
        colmin=min(source_val);
      elseif isequal(climPolicy, 'sym')
        colmax=source_val_max;
        colmin=-source_val_max;
      end
      
      if ~isfield(para, 'colormap')
        if isequal(climPolicy, 'minmax') || sign(colmax) == sign(colmin) || abs(colmin) < eps
          load('cm13', 'cm13')
          mycolormap= cm13(32:end, :);
        else
          load('cm13', 'cm13')
          mycolormap= cm13;
        end
      else
        mycolormap = para.colormap;
      end
      
      for i=1:np;
        source_x(i, 4) = max(source_x(i, 4), colmin);
        source_x(i, 4) = min(source_x(i, 4), colmax);
        colorloc = mycolormap(min(floor((size(mycolormap, 1)-1)*((source_x(i, 4)-colmin)/(colmax-colmin)))+1, size(mycolormap, 1)), :);
        points_loc=source_x(i,1:3)+fsv*myviewdir;
        [xs,ys,zs] = sphere;
        hs = surf((xs*mymarkersize/3)+points_loc(1, 1),(ys*mymarkersize/3)+points_loc(1, 2),(zs*mymarkersize/3)+points_loc(1, 3));
        set(hs, 'facecolor', colorloc, 'edgecolor', 'none')

        if isequal(dipnames, 1)
          points_loc=source_x(i,1:3)+fsv2*myviewdir;
          ht = text(points_loc(1, 1)-1, points_loc(1, 2)+1, points_loc(1, 3)-1, num2str(i), 'fontsize', myfontsize, 'fontweight', 'bold');
          set(ht, 'color', colorloc, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle')
        elseif iscell(dipnames)
          points_loc=source_x(i,1:3)+fsv2*myviewdir;
          ht = text(points_loc(1, 1)-1, points_loc(1, 2)+1, points_loc(1, 3)-1, dipnames{k}{i}, 'fontsize', myfontsize, 'fontweight', 'bold');
          set(ht, 'color', colorloc, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle')
        end
      end
      
      if dipcolorbars==1  
        colormap(mycolormap)
        pos=get(gca,'pos');
        set(gca,'pos',[pos(1) pos(2) 0.8*pos(3) pos(4)]);
        pos=get(gca,'pos');
        han.dipcb = colorbar('location','eastoutside', 'position', [pos(1)+pos(3)+0.1 pos(2)+0.05 0.05 pos(4)-0.1]);
        caxis([colmin colmax])
      end
    elseif ndum==6
       for i=1:np;
          if strcmp(dipcolorstyle,'mixed')
              colorloc=colors{i};
          else
              colorloc=colors{k};
          end
          points_loc=source_x(i,1:3)+fsv*myviewdir;
          ori_loc=source_x(i,4:6);
          ori_loc_norm=ori_loc/norm(ori_loc);
          if para.normalize==1
             pp=[points_loc;points_loc+dipscal*ori_loc_norm];
          else
             pp=[points_loc;points_loc+ori_loc];
          end
          [xs,ys,zs] = sphere;
          hs = surf((xs*mymarkersize/2)+pp(1, 1),(ys*mymarkersize/2)+pp(1, 2),(zs*mymarkersize/2)+pp(1, 3));
          set(hs, 'facecolor', colorloc, 'edgecolor', 'none')
          [Cylinder EndPlate1 EndPlate2] = cyl(pp(1, :),pp(2, :),10/4,20,colorloc,0,0);
%           [xc, yc, zc] = cylinder(ori_loc);
%           hs = surf((xc*mymarkersize/4)+pp(1, 1),(yc*mymarkersize/4)+pp(1, 2),2*(zc*mymarkersize)+pp(1, 3));
          set(Cylinder, 'facecolor', colorloc, 'edgecolor', 'none')
          
          if isequal(dipnames, 1)
              points_loc=source_x(i,1:3)+fsv2*myviewdir;
              ht = text(points_loc(1, 1)-1, points_loc(1, 2)+1, points_loc(1, 3)-1, num2str(i), 'fontsize', myfontsize, 'fontweight', 'bold');
              set(ht, 'color', colorloc, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle')
          elseif iscell(dipnames)
              points_loc=source_x(i,1:3)+fsv2*myviewdir;
              ht = text(points_loc(1, 1)-1, points_loc(1, 2)+1, points_loc(1, 3)-1, dipnames{k}{i}, 'fontsize', myfontsize, 'fontweight', 'bold');
              set(ht, 'color', colorloc, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle')
          end
       end
    end
end

% keyboard

% axis(figscalevec);
axis tight
axis equal

if isfield(para,'title');
  htit = text((maxs(1)+mins(1))/2, mins(2)-1, maxs(3)+0.5, para.title, 'fontsize', ceil(mydirfontsize/2), 'fontweight', 'bold', 'color', 'k');
  set(htit, 'horizontalalignment', 'left', 'verticalalignment', 'middle')
  
end
set(gca,'visible','off')

save varargin.mat varargin;
   
return;
